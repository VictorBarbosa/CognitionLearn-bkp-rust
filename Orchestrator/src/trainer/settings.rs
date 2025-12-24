use serde::{Deserialize, Serialize};
use crate::agent::AgentType;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainerSettings {
    pub algorithm: AgentType,
    pub batch_size: usize,
    pub buffer_size: usize,
    pub learning_rate: f32,
    pub hidden_units: usize,
    pub summary_freq: usize,
    pub checkpoint_interval: usize,
    pub keep_checkpoints: usize, // Added
    pub max_steps: usize,
    pub resume: bool,
    pub epsilon: Option<f32>, // for PPO
    pub tau: Option<f32>,     // for SAC/TD3
    pub gamma: f32,
    pub num_epochs: usize,    // PPO/POCA epochs
    pub lambd: Option<f32>,   // GAE Lambda (PPO/POCA)
    pub entropy_coef: Option<f32>, // Entropy Coefficient (PPO/POCA/SAC init)
    pub policy_delay: Option<usize>, // TD3/TDSAC
    pub n_quantiles: Option<usize>, // TQC
    pub n_to_drop: Option<usize>, // TQC
    
    pub curiosity_strength: Option<f32>, // PPO_CE
    pub curiosity_learning_rate: Option<f32>, // PPO_CE
    
    // Memory (LSTM)
    pub memory_size: Option<usize>,
    pub sequence_length: Option<usize>,

    pub show_obs: bool,
    pub output_path: String,
    pub init_path: String,
    pub device: String, // Added
}

impl Default for TrainerSettings {
    fn default() -> Self {
        Self {
            algorithm: AgentType::SAC,
            batch_size: 256,
            buffer_size: 1000000,
            learning_rate: 3e-4,
            hidden_units: 256,
            summary_freq: 1000,
            checkpoint_interval: 5000,
            keep_checkpoints: 5, // Default
            max_steps: 1000000,
            resume: false,
            epsilon: Some(0.2),
            tau: Some(0.005),
            gamma: 0.99,
            num_epochs: 3,
            lambd: Some(0.95),
            entropy_coef: Some(0.01),
            policy_delay: Some(2),
            n_quantiles: Some(25),
            n_to_drop: Some(2),
            curiosity_strength: None,
            curiosity_learning_rate: None,
            memory_size: None,
            sequence_length: None,
            show_obs: false,
            output_path: "".to_string(),
            init_path: "".to_string(),
            device: "cpu".to_string(),
        }
    }
}
