use memmap2::MmapMut;
use std::fmt;
use std::fs::OpenOptions;
use std::io::{self};
use std::time::{Duration, Instant};
use std::{thread, fs};

// --- Error Type ---
#[derive(Debug)]
pub enum ChannelError {
    Io(io::Error),
    StringConversion(std::string::FromUtf8Error),
    SendError(String),
    ReceiveError(String),
}

impl From<io::Error> for ChannelError {
    fn from(err: io::Error) -> Self {
        ChannelError::Io(err)
    }
}

impl From<std::string::FromUtf8Error> for ChannelError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        ChannelError::StringConversion(err)
    }
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChannelError::Io(err) => write!(f, "IO error: {}", err),
            ChannelError::StringConversion(err) => write!(f, "String conversion error: {}", err),
            ChannelError::SendError(msg) => write!(f, "Send error: {}", msg),
            ChannelError::ReceiveError(msg) => write!(f, "Receive error: {}", msg),
        }
    }
}

impl std::error::Error for ChannelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ChannelError::Io(err) => Some(err),
            ChannelError::StringConversion(err) => Some(err),
            _ => None,
        }
    }
}

// --- SharedMemoryBuffer with Status Flag ---
// Protocol:
// Byte 0: Status Flag (0 = Empty/Read, 1 = Ready/Written)
// Byte 1..N: Data payload
#[derive(Debug)]
struct SharedMemoryBuffer {
    name: String,
    size: u64,
    mmap: MmapMut,
}

impl SharedMemoryBuffer {
    fn create(name: String, size: u64) -> Result<Self, io::Error> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&name)?;
        file.set_len(size)?;
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        
        // Initialize flag to 0 (Empty)
        mmap[0] = 0;
        
        Ok(SharedMemoryBuffer { name, size, mmap })
    }

    // Write data and set flag to 1
    fn write_with_signal(&mut self, data: &[u8]) -> Result<(), String> {
        // Offset 1 for data, leaving byte 0 for flag
        if (data.len() + 1) as u64 > self.size {
            return Err(format!(
                "Data size ({} bytes) is too large for the buffer ({} bytes).",
                data.len(),
                self.size
            ));
        }
        
        // Wait if buffer is not empty (Backpressure/Safety)
        // In a strict Request-Reply loop, this shouldn't block long, but good for safety.
        // For simple Producer-Consumer, we might overwrite, but let's be safe.
        // NOTE: For this specific high-perf implementation, we assume caller controls flow.
        
        // 1. Write Data at Offset 1
        self.mmap[1..1 + data.len()].copy_from_slice(data);
        self.mmap[1 + data.len()] = 0; // Null terminator for safety
        
        // 2. Set Flag to 1 (Signal Ready)
        // Using volatile write to ensure ordering if compiler tries to be smart, 
        // though standard write is usually fine on x86. Rust's volatile is for MMIO mostly.
        // Simple assignment is sufficient here as mmap is volatile storage.
        self.mmap[0] = 1;
        self.mmap.flush().map_err(|e| e.to_string())?; // Ensure flush
        
        Ok(())
    }

    // Wait for flag == 1, Read data, then set flag = 0
    fn read_with_wait(&mut self, timeout_ms: u64) -> Result<Vec<u8>, String> {
        let start = Instant::now();
        let timeout = Duration::from_millis(timeout_ms);
        let mut sleep_duration = Duration::from_micros(50); // Start fast

        // Spin-wait loop (Polite)
        while self.mmap[0] == 0 {
            if start.elapsed() > timeout {
                return Err(format!("Timeout waiting for data in: {}", self.name));
            }
            thread::sleep(sleep_duration);
            // Exponential backoff up to 1ms
            if sleep_duration < Duration::from_millis(1) {
                sleep_duration = sleep_duration.mul_f32(1.5);
            }
        }

        // Data is ready (Flag == 1)
        // Find end of data (null terminator) starting from offset 1
        let end = self.mmap[1..]
            .iter()
            .position(|&b| b == 0)
            .map(|pos| pos + 1) // Adjust for slice offset
            .unwrap_or(self.mmap.len());
        
        let data = self.mmap[1..end].to_vec();

        // Reset Flag to 0 (Ack)
        self.mmap[0] = 0;
        // self.mmap.flush().ok(); // Optional flush, OS handles coherence usually

        Ok(data)
    }
    
    // Check if data is ready without blocking
    fn has_data(&self) -> bool {
        self.mmap[0] == 1
    }
}

// --- CommunicationPipe (Wrapper) ---
#[derive(Debug)]
struct CommunicationPipe {
    buffer: SharedMemoryBuffer,
}

impl CommunicationPipe {
    fn create(base_name: String, size: u64) -> Result<Self, io::Error> {
        let buffer_name = format!("{}.mem", base_name);
        Ok(CommunicationPipe {
            buffer: SharedMemoryBuffer::create(buffer_name, size)?,
        })
    }

    fn send(&mut self, data: &[u8]) -> Result<(), ChannelError> {
        self.buffer.write_with_signal(data).map_err(ChannelError::SendError)
    }

    fn receive(&mut self, timeout_ms: u64) -> Result<Vec<u8>, ChannelError> {
        self.buffer.read_with_wait(timeout_ms).map_err(ChannelError::ReceiveError)
    }

    fn has_msg(&self) -> bool {
        self.buffer.has_data()
    }
}

// --- CommunicationChannel ---
#[derive(Debug)]
#[allow(dead_code)]
pub struct CommunicationChannel {
    channel_id: String,
    forward_pipe: CommunicationPipe,
    backward_pipe: CommunicationPipe,
}

impl CommunicationChannel {
    pub fn create(channel_id: &str, size: u64, base_path: &str) -> Result<Self, io::Error> {
        let base_path = std::path::Path::new(base_path);
        
        if !base_path.exists() {
            let _ = std::fs::create_dir_all(&base_path);
        }

        let fwd_base = base_path.join(format!("cognition_fwd_{}", channel_id));
        let bwd_base = base_path.join(format!("cognition_bwd_{}", channel_id));
        
        Ok(CommunicationChannel {
            channel_id: channel_id.to_string(),
            forward_pipe: CommunicationPipe::create(fwd_base.to_str().unwrap().to_string(), size)?,
            backward_pipe: CommunicationPipe::create(bwd_base.to_str().unwrap().to_string(), size)?,
        })
    }
    
    // For a client (like Unity)
    #[allow(dead_code)]
    pub fn request(&mut self, data: &[u8], timeout_ms: u64) -> Result<Vec<u8>, ChannelError> {
        self.forward_pipe.send(data)?;
        self.backward_pipe.receive(timeout_ms)
    }

    // For a server (like the Rust orchestrator)
    pub fn listen<F>(&mut self, mut handler: F, timeout_ms: u64) -> Result<(), ChannelError>
    where
        F: FnMut(Vec<u8>) -> Result<Vec<u8>, String>,
    {
        let received_data = self.forward_pipe.receive(timeout_ms)?;
        match handler(received_data) {
            Ok(response_data) => {
                self.backward_pipe.send(&response_data)?;
                Ok(())
            }
            Err(e) => Err(ChannelError::SendError(format!("Handler error: {}", e))),
        }
    }

    pub fn has_msg(&self) -> bool {
        self.forward_pipe.has_msg()
    }
}
