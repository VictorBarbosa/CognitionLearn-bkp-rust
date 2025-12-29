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
        private readonly long _capacity;

        public SharedMemoryBuffer(string name, long size)
        {
            _name = name;
            _capacity = size;
            // This approach of creating a FileStream is crucial for cross-platform compatibility, especially on macOS.
            // Ensure the file has the correct size
            var fileStream = new FileStream(name, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite);
            fileStream.SetLength(size); 
            _mmf = MemoryMappedFile.CreateFromFile(fileStream, null, size, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, false);
            _accessor = _mmf.CreateViewAccessor();
        }

        public byte ReadFlag()
        {
            return _accessor.ReadByte(0);
        }

        public void WriteFlag(byte value)
        {
            _accessor.Write(0, value);
        }

        public void WriteData(byte[] data)
        {
            // Data starts at offset 1
            if (data.Length + 1 >= _accessor.Capacity)
            {
                throw new ArgumentOutOfRangeException(nameof(data), "Data is too large for the shared memory buffer.");
            }
            _accessor.WriteArray(1, data, 0, data.Length);
            _accessor.Write(1 + data.Length, (byte)0); // Null terminator
        }

        public byte[] ReadData()
        {
            // Data starts at offset 1
            var buffer = new byte[_accessor.Capacity - 1];
            _accessor.ReadArray(1, buffer, 0, buffer.Length);
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

    // --- 2.3. CommunicationPipe ---
    internal class CommunicationPipe : IDisposable
    {
        private readonly SharedMemoryBuffer _buffer;

        public CommunicationPipe(string baseName, long size)
        {
            _buffer = new SharedMemoryBuffer($"{baseName}.mem", size);
        }

        public void Send(byte[] data)
        {
            // 1. Backpressure: Wait for the other side to acknowledge (Flag == 0)
            int safetyCounter = 0;
            while (_buffer.ReadFlag() == 1)
            {
                Thread.Sleep(1); // Sleep 1ms to release CPU
                safetyCounter++;
                if (safetyCounter > 5000)
                {
                    Debug.LogWarning("Orchestrator took too long to read (Send Timeout). Resetting flag.");
                    break;
                }
            }

            // 2. Write data (Offset 1)
            _buffer.WriteData(data);

            // 3. Signal Data Ready (Flag == 1)
            _buffer.WriteFlag(1);
        }

        public byte[] Receive(int timeoutMs)
        {
            // 1. Wait for data (Flag == 1)
            int waited = 0;
            while (_buffer.ReadFlag() == 0)
            {
                if (waited >= timeoutMs)
                {
                    return null; // Timeout
                }
                Thread.Sleep(1);
                waited++;
            }

            // 2. Read data
            var result = _buffer.ReadData();

            // 3. Acknowledge Receipt (Flag == 0)
            _buffer.WriteFlag(0);

            return result;
        }

        public async Task<byte[]> ReceiveAsync(int timeoutMs)
        {
            // 1. Wait for data (Flag == 1)
            int waited = 0;
            while (_buffer.ReadFlag() == 0)
            {
                if (waited >= timeoutMs)
                {
                    return null; // Timeout
                }
                await Task.Delay(1);
                waited++;
            }

            // 2. Read data
            var result = _buffer.ReadData();

            // 3. Acknowledge Receipt (Flag == 0)
            _buffer.WriteFlag(0);

            return result;
        }

        public void Dispose()
        {
            _buffer?.Dispose();
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