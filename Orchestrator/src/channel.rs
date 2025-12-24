use memmap2::MmapMut;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self}; // Removed Read
use std::time::Duration;
use std::{fs, thread};

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

// --- 2.1. SharedMemoryBuffer ---
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
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(SharedMemoryBuffer { name, size, mmap })
    }

    fn write(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() as u64 >= self.size {
            return Err(format!(
                "Data size ({} bytes) is too large for the buffer ({} bytes).",
                data.len(),
                self.size
            ));
        }
        self.mmap[..data.len()].copy_from_slice(data);
        self.mmap[data.len()] = 0; // Null terminator for string compatibility
        Ok(())
    }

    fn read(&self) -> Result<Vec<u8>, String> {
        let end = self
            .mmap
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.mmap.len());
        Ok(self.mmap[..end].to_vec())
    }
}

// Drop implementation removed to prevent premature deletion of memory files.
// Files should be managed by higher-level logic (e.g., overwritten or explicitly cleaned).
// impl Drop for SharedMemoryBuffer {
//     fn drop(&mut self) {
//         let _ = fs::remove_file(&self.name);
//     }
// }


// --- 2.2. SyncPrimitive ---
#[derive(Debug)]
struct SyncPrimitive {
    name: String,
}

impl SyncPrimitive {
    fn create(name: String) -> Self {
        // Ensure file doesn't exist initially
        let _ = fs::remove_file(&name);
        SyncPrimitive { name }
    }

    fn signal(&self) -> Result<(), io::Error> {
        File::create(&self.name)?;
        Ok(())
    }

    fn wait(&self, timeout_ms: u64) -> Result<(), String> {
        let sleep_duration = Duration::from_micros(10);
        let mut total_elapsed = 0;

        while fs::metadata(&self.name).is_err() {
            thread::sleep(sleep_duration);
            // We are checking every 10us approx.
            // total_elapsed is roughly in "units of loops".
            // To respect timeout_ms, we need to convert logic.
            // 1ms = 1000us. 100 loops of 10us = 1ms.
            total_elapsed += 1;
            if total_elapsed * 10 > timeout_ms * 1000 {
                return Err(format!("Timeout waiting for signal: {}", self.name));
            }
        }
        fs::remove_file(&self.name).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn check(&self) -> bool {
        fs::metadata(&self.name).is_ok()
    }
}

// Drop implementation removed to prevent premature deletion of signal files.
// Files should be consumed/deleted by the Reader (`wait` method).
// impl Drop for SyncPrimitive {
//     fn drop(&mut self) {
//         let _ = fs::remove_file(&self.name);
//     }
// }


// --- 2.3. CommunicationPipe ---
#[derive(Debug)]
struct CommunicationPipe {
    buffer: SharedMemoryBuffer,
    signal: SyncPrimitive,
}

impl CommunicationPipe {
    fn create(base_name: String, size: u64) -> Result<Self, io::Error> {
        let buffer_name = format!("{}.mem", base_name);
        let signal_name = format!("{}.sig", base_name);

        Ok(CommunicationPipe {
            buffer: SharedMemoryBuffer::create(buffer_name, size)?,
            signal: SyncPrimitive::create(signal_name),
        })
    }

    fn send(&mut self, data: &[u8]) -> Result<(), ChannelError> {
        self.buffer
            .write(data)
            .map_err(ChannelError::SendError)?;
        self.signal.signal()?;
        Ok(())
    }

    fn receive(&self, timeout_ms: u64) -> Result<Vec<u8>, ChannelError> {
        self.signal.wait(timeout_ms).map_err(ChannelError::ReceiveError)?;
        self.buffer.read().map_err(ChannelError::ReceiveError)
    }

    fn has_msg(&self) -> bool {
        self.signal.check()
    }
}

// --- 2.4. CommunicationChannel ---
#[derive(Debug)]
#[allow(dead_code)]
pub struct CommunicationChannel {
    channel_id: String,
    forward_pipe: CommunicationPipe,
    backward_pipe: CommunicationPipe,
}

impl CommunicationChannel {
    pub fn create(channel_id: &str, size: u64) -> Result<Self, io::Error> {
        let mut base_path = std::env::temp_dir();
        base_path.push("cognition_memory");
        
        // Ensure directory exists
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
