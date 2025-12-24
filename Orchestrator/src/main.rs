mod channel;
mod sac;
mod ppo;
mod onnx_utils;
mod td3;
mod bc;
mod poca;
mod tdsac;
mod tqc;
mod crossq;
mod agent;
mod protocol;
mod onnx; // Generated protobuf module
mod trainer;
mod gui;
mod app;
mod pages;

use clap::Parser;
// use crate::app::MyApp;
// use crate::trainer::Trainer; // Will need to reintegrate later
// use crate::trainer::settings::TrainerSettings;
// use crate::agent::AgentType;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Algorithm to use (sac, ppo, td3, bc)
    #[arg(short, long, default_value = "ppo")]
    algo: String,

    /// Number of hidden units in the networks
    #[arg(long, default_value_t = 256)]
    hidden_units: usize,

    /// Batch size for training
    #[arg(short, long, default_value_t = 256)]
    batch_size: usize,

    /// Replay buffer size
    #[arg(long, default_value_t = 1000000)]
    buffer_size: usize,

    /// Learning rate
    #[arg(short, long, default_value_t = 3e-4)]
    lr: f32,

    /// Summary frequency (steps)
    #[arg(long, default_value_t = 1000)]
    summary_freq: usize,

    /// Save frequency (steps)
    #[arg(long, default_value_t = 5000)]
    checkpoint_interval: usize,

    /// Max training steps
    #[arg(long, default_value_t = 10000000)]
    max_steps: usize,

    /// Resume from checkpoint
    #[arg(long, default_value_t = false)]
    resume: bool,

    /// Show raw observations logs
    #[arg(long, default_value_t = false)]
    show_obs: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 CognitionLearn Orchestrator - Start Screen");
    
    // Diagnostic Logs
    println!("--- Hardware Diagnostics ---");
    println!("  CUDA Available: {}", tch::Cuda::is_available());
    println!("  MPS Available:  {}", tch::utils::has_mps());
    println!("  Cudnn Available: {}", tch::Cuda::cudnn_is_available());
    println!("  Device Count:   {}", tch::Cuda::device_count());
    println!("----------------------------");

    // Launch the GUI (which is exactly the reference code)
    if let Err(e) = gui::run() {
        eprintln!("GUI Error: {}", e);
    }
    Ok(())
}