using System;

namespace Unity.CognitionLearn
{
    /// <summary>
    /// Minimal abstraction over a named shared memory region.
    /// Implementations differ per OS (Windows uses MemoryMappedFile, Unix uses POSIX shm_open).
    /// </summary>
    public interface ISharedMemory : IDisposable
    {
        /// <summary>Write a byte array at the given offset.</summary>
        void Write(byte[] data, int offset);
        /// <summary>Read a byte array of length from the given offset.</summary>
        byte[] Read(int offset, int length);
        /// <summary>Write a 32‑bit integer at the given offset.</summary>
        void WriteInt(int value, int offset);
        /// <summary>Read a 32‑bit integer from the given offset.</summary>
        int ReadInt(int offset);
        /// <summary>Total size of the shared memory region (bytes).</summary>
        long Size { get; }
    }
}
