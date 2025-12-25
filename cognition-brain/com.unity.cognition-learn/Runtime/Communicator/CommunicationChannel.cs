using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Unity.CognitionLearn.Communicator
{
    // --- 2.1. SharedMemoryBuffer ---
    internal class SharedMemoryBuffer : IDisposable
    {
        private readonly string _name;
        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _accessor;

        public SharedMemoryBuffer(string name, long size)
        {
            _name = name;
            // This approach of creating a FileStream is crucial for cross-platform compatibility, especially on macOS.
            var fileStream = new FileStream(name, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite);
            fileStream.SetLength(size); // Ensure the file has the correct size
            _mmf = MemoryMappedFile.CreateFromFile(fileStream, null, size, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, false);
            _accessor = _mmf.CreateViewAccessor();
        }

        public void Write(byte[] data)
        {
            if (data.Length >= _accessor.Capacity)
            {
                throw new ArgumentOutOfRangeException(nameof(data), "Data is too large for the shared memory buffer.");
            }
            _accessor.WriteArray(0, data, 0, data.Length);
            _accessor.Write(data.Length, (byte)0); // Null terminator
        }

        public byte[] Read()
        {
            var buffer = new byte[_accessor.Capacity];
            _accessor.ReadArray(0, buffer, 0, buffer.Length);
            int end = Array.IndexOf(buffer, (byte)0);
            if (end == -1) end = buffer.Length;

            var result = new byte[end];
            Array.Copy(buffer, result, end);
            return result;
        }

        public void Dispose()
        {
            _accessor?.Dispose();
            _mmf?.Dispose();
            try
            {
                if (File.Exists(_name))
                {
                    File.Delete(_name);
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Could not delete memory buffer file {_name}: {e.Message}");
            }
        }
    }

    // --- 2.2. SyncPrimitive ---
    internal class SyncPrimitive : IDisposable
    {
        private readonly string _name;

        public SyncPrimitive(string name)
        {
            _name = name;
            // Clean up any old signal files on creation
            if (File.Exists(_name))
            {
                File.Delete(_name);
            }
        }

        public void Signal()
        {
            try
            {
                File.Create(_name).Close();
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Could not create signal file {_name}. This might be a permissions issue. Error: {e.Message}");
            }
        }

        public void Wait(int timeoutMs)
        {
            // Blocking wait (Legacy/Sync) optimized for high throughput
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            
            // Initial spin wait for very fast response (sub-millisecond)
            int spinCount = 0;
            while (!File.Exists(_name))
            {
                if (spinCount < 1000)
                {
                    Thread.SpinWait(10); // Short busy wait
                    spinCount++;
                }
                else
                {
                    // Yield to other threads if it takes longer, but don't sleep for full 1ms
                    Thread.Yield(); 
                }

                if (stopwatch.ElapsedMilliseconds > timeoutMs)
                {
                    throw new TimeoutException($"Timed out waiting for signal file: {_name}");
                }
            }
            File.Delete(_name);
        }

        public async Task WaitAsync(int timeoutMs)
        {
            // Non-blocking wait (Async)
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            while (!File.Exists(_name))
            {
                // Yield control to avoiding blocking a Thread pool thread with Sleep
                await Task.Delay(10); 
                
                if (stopwatch.ElapsedMilliseconds > timeoutMs)
                {
                    throw new TimeoutException($"Timed out waiting for signal file: {_name}");
                }
            }
            
            // Try to delete. If it fails (racing?), it's usually fine as we consumed the signal.
            try 
            {
                File.Delete(_name);
            }
            catch (IOException) 
            { 
                 // Ignore deletion errors in async race, or retry
            }
        }

        public void Dispose()
        {
            try
            {
                if (File.Exists(_name))
                {
                    File.Delete(_name);
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Could not delete signal file {_name}: {e.Message}");
            }
        }
    }

    // --- 2.3. CommunicationPipe ---
    internal class CommunicationPipe : IDisposable
    {
        private readonly SharedMemoryBuffer _buffer;
        private readonly SyncPrimitive _signal;

        public CommunicationPipe(string baseName, long size)
        {
            _buffer = new SharedMemoryBuffer($"{baseName}.mem", size);
            _signal = new SyncPrimitive($"{baseName}.sig");
        }

        public void Send(byte[] data)
        {
            _buffer.Write(data);
            _signal.Signal();
        }

        public byte[] Receive(int timeoutMs)
        {
            _signal.Wait(timeoutMs);
            return _buffer.Read();
        }

        public async Task<byte[]> ReceiveAsync(int timeoutMs)
        {
            await _signal.WaitAsync(timeoutMs);
            return _buffer.Read();
        }

        public void Dispose()
        {
            _buffer?.Dispose();
            _signal?.Dispose();
        }
    }

    // --- 2.4. CommunicationChannel ---
    internal class CommunicationChannel : IDisposable
    {
        private readonly CommunicationPipe _forwardPipe;
        private readonly CommunicationPipe _backwardPipe;

        public CommunicationChannel(string channelId, long size, string customPath = null)
        {
            string tempPath;
            if (string.IsNullOrEmpty(customPath))
            {
                tempPath = Path.Combine(Path.GetTempPath(), "cognition_memory");
            }
            else
            {
                tempPath = customPath;
            }
            
            if (!Directory.Exists(tempPath))
            {
                Directory.CreateDirectory(tempPath);
            }

            var fwdBasePath = Path.Combine(tempPath, $"cognition_fwd_{channelId}");
            var bwdBasePath = Path.Combine(tempPath, $"cognition_bwd_{channelId}");
            
            _forwardPipe = new CommunicationPipe(fwdBasePath, size);
            _backwardPipe = new CommunicationPipe(bwdBasePath, size);
        }
        
        public byte[] Request(byte[] data, int timeoutMs = 60000)
        {
            _forwardPipe.Send(data);
            return _backwardPipe.Receive(timeoutMs);
        }

        public async Task<byte[]> RequestAsync(byte[] data, int timeoutMs = 60000)
        {
            _forwardPipe.Send(data);
            return await _backwardPipe.ReceiveAsync(timeoutMs);
        }

        public void Dispose()
        {
            _forwardPipe?.Dispose();
            _backwardPipe?.Dispose();
        }
    }
}
