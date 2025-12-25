
// Configuration enums and structs

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ConfigSection {
    Settings,
    Algorithms,
}

#[derive(Debug, PartialEq, Clone, Copy)] // Added Clone, Copy to use with radio buttons
pub enum CheckpointMode {
    None,
    LoadModel,
    Resume,
    Force,
    TrainModel,
    Inference,
}

#[derive(Debug)]
pub struct EngineSettings {
    pub width: u16,
    pub height: u16,
    pub quality_level: u8,
    pub time_scale: f32,
    pub target_frame_rate: u16,
    pub capture_frame_rate: u16,
    pub no_graphics: bool,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            width: 250,
            height: 250,
            quality_level: 1, // Reduced quality for headless performance
            time_scale: 20.0, // Reduced time scale to prevent CPU saturation
            target_frame_rate: 60,
            capture_frame_rate: 0,
            no_graphics: true,
        }
    }
}

#[derive(Debug)]
pub struct CheckpointSettings {
    pub run_id: String,
    pub mode: CheckpointMode, // Substituído os booleanos por um enum
    pub results_dir: String,
    pub enable_race_mode: bool,
}

impl Default for CheckpointSettings {
    fn default() -> Self {
        Self {
            run_id: String::from(""),
            mode: CheckpointMode::None,
            results_dir: String::from("results"),
            enable_race_mode: false,
        }
    }
}


#[derive(Debug, PartialEq, Clone, Copy)]
pub enum AlgorithmConfigStep {
    Selection,
    Configuration,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum AlgorithmSelectionMode {
    None, // No mode selected initially
    Same, // Select only one algorithm
    Different, // Select multiple algorithms
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AlgoConfig {
    pub trainer_type: String,
    pub batch_size: u32,
    pub buffer_size: u32,
    pub buffer_init_steps: u32,
    pub tau: f32,
    pub steps_per_update: f32,
    pub save_replay_buffer: bool,
    pub init_entcoef: Option<f32>,
    pub beta: Option<f32>,
    pub epsilon: Option<f32>,
    pub lambd: Option<f32>,
    pub num_epoch: Option<u32>,
    pub policy_delay: Option<u32>,
    pub n_quantiles: Option<u32>,
    pub n_to_drop: Option<u32>,
    pub destructive_threshold: Option<f32>,
    pub image_pad: Option<u32>,
    pub entropy_temperature: Option<f32>,
    pub adaptive_entropy_temperature: Option<bool>,
    pub curiosity_strength: Option<f32>,
    pub curiosity_gamma: Option<f32>,
    pub curiosity_learning_rate: Option<f32>,
    pub curiosity_hidden_units: Option<u32>,
    pub curiosity_num_layers: Option<u32>,
    pub imagination_horizon: Option<u32>,
    pub use_imagination_augmented: Option<bool>,
    
    // Memory
    pub use_memory: bool,
    pub memory_sequence_length: Option<u32>,
    pub memory_size: Option<u32>,
    
    pub time_horizon: u32,
}

impl AlgoConfig {
    pub fn ppo() -> Self { Self { trainer_type: "ppo".into(), batch_size: 2048, buffer_size: 20480, buffer_init_steps: 0, tau: 0.95, steps_per_update: 1.0, save_replay_buffer: false, init_entcoef: Some(0.005), beta: Some(0.005), epsilon: Some(0.2), lambd: Some(0.95), num_epoch: Some(3), policy_delay: None, n_quantiles: None, n_to_drop: None, destructive_threshold: None, image_pad: None, entropy_temperature: None, adaptive_entropy_temperature: None, curiosity_strength: None, curiosity_gamma: None, curiosity_learning_rate: None, curiosity_hidden_units: None, curiosity_num_layers: None, imagination_horizon: None, use_imagination_augmented: None, use_memory: false, memory_sequence_length: Some(64), memory_size: Some(256), time_horizon: 1000 } }
    pub fn sac() -> Self { Self { trainer_type: "sac".into(), batch_size: 1024, buffer_size: 300000, buffer_init_steps: 5000, tau: 0.005, steps_per_update: 1.0, save_replay_buffer: true, init_entcoef: Some(0.5), beta: None, epsilon: None, lambd: None, num_epoch: None, policy_delay: None, n_quantiles: None, n_to_drop: None, destructive_threshold: None, image_pad: None, entropy_temperature: None, adaptive_entropy_temperature: Some(true), curiosity_strength: None, curiosity_gamma: None, curiosity_learning_rate: None, curiosity_hidden_units: None, curiosity_num_layers: None, imagination_horizon: None, use_imagination_augmented: None, use_memory: false, memory_sequence_length: Some(64), memory_size: Some(256), time_horizon: 1000 } }
    pub fn td3() -> Self { Self { trainer_type: "td3".into(), batch_size: 1024, buffer_size: 300000, buffer_init_steps: 5000, tau: 0.005, steps_per_update: 1.0, save_replay_buffer: true, init_entcoef: None, beta: None, epsilon: None, lambd: None, num_epoch: None, policy_delay: Some(2), n_quantiles: None, n_to_drop: None, destructive_threshold: None, image_pad: None, entropy_temperature: None, adaptive_entropy_temperature: None, curiosity_strength: None, curiosity_gamma: None, curiosity_learning_rate: None, curiosity_hidden_units: None, curiosity_num_layers: None, imagination_horizon: None, use_imagination_augmented: None, use_memory: false, memory_sequence_length: Some(64), memory_size: Some(256), time_horizon: 1000 } }
    pub fn tdsac() -> Self { Self { trainer_type: "tdsac".into(), batch_size: 1024, buffer_size: 300000, buffer_init_steps: 5000, tau: 0.005, steps_per_update: 1.0, save_replay_buffer: true, init_entcoef: Some(0.5), beta: None, epsilon: None, lambd: None, num_epoch: None, policy_delay: Some(2), n_quantiles: None, n_to_drop: None, destructive_threshold: None, image_pad: None, entropy_temperature: None, adaptive_entropy_temperature: Some(true), curiosity_strength: None, curiosity_gamma: None, curiosity_learning_rate: None, curiosity_hidden_units: None, curiosity_num_layers: None, imagination_horizon: None, use_imagination_augmented: None, use_memory: false, memory_sequence_length: Some(64), memory_size: Some(256), time_horizon: 1000 } }
    pub fn tqc() -> Self { Self { trainer_type: "tqc".into(), batch_size: 1024, buffer_size: 300000, buffer_init_steps: 5000, tau: 0.005, steps_per_update: 1.0, save_replay_buffer: true, init_entcoef: Some(0.5), beta: None, epsilon: None, lambd: None, num_epoch: None, policy_delay: None, n_quantiles: Some(25), n_to_drop: Some(2), destructive_threshold: None, image_pad: None, entropy_temperature: None, adaptive_entropy_temperature: Some(true), curiosity_strength: None, curiosity_gamma: None, curiosity_learning_rate: None, curiosity_hidden_units: None, curiosity_num_layers: None, imagination_horizon: None, use_imagination_augmented: None, use_memory: false, memory_sequence_length: Some(64), memory_size: Some(256), time_horizon: 1000 } }
    pub fn crossq() -> Self { Self { trainer_type: "crossq".into(), batch_size: 1024, buffer_size: 300000, buffer_init_steps: 5000, tau: 0.005, steps_per_update: 1.0, save_replay_buffer: true, init_entcoef: None, beta: None, epsilon: None, lambd: None, num_epoch: None, policy_delay: None, n_quantiles: None, n_to_drop: None, destructive_threshold: None, image_pad: None, entropy_temperature: None, adaptive_entropy_temperature: Some(true), curiosity_strength: None, curiosity_gamma: None, curiosity_learning_rate: None, curiosity_hidden_units: None, curiosity_num_layers: None, imagination_horizon: None, use_imagination_augmented: None, use_memory: false, memory_sequence_length: Some(64), memory_size: Some(256), time_horizon: 1000 } }
    pub fn drqv2() -> Self { Self { trainer_type: "drqv2".into(), batch_size: 256, buffer_size: 100000, buffer_init_steps: 1000, tau: 0.005, steps_per_update: 1.0, save_replay_buffer: true, init_entcoef: Some(0.1), beta: None, epsilon: None, lambd: None, num_epoch: None, policy_delay: None, n_quantiles: None, n_to_drop: None, destructive_threshold: None, image_pad: Some(4), entropy_temperature: None, adaptive_entropy_temperature: Some(true), curiosity_strength: None, curiosity_gamma: None, curiosity_learning_rate: None, curiosity_hidden_units: None, curiosity_num_layers: None, imagination_horizon: None, use_imagination_augmented: None, use_memory: false, memory_sequence_length: Some(64), memory_size: Some(256), time_horizon: 1000 } }
    pub fn ppo_et() -> Self { Self { trainer_type: "ppo_et".into(), batch_size: 2048, buffer_size: 20480, buffer_init_steps: 0, tau: 0.95, steps_per_update: 1.0, save_replay_buffer: false, init_entcoef: None, beta: Some(0.005), epsilon: Some(0.2), lambd: Some(0.95), num_epoch: Some(3), policy_delay: None, n_quantiles: None, n_to_drop: None, destructive_threshold: None, image_pad: None, entropy_temperature: Some(0.1), adaptive_entropy_temperature: Some(true), curiosity_strength: None, curiosity_gamma: None, curiosity_learning_rate: None, curiosity_hidden_units: None, curiosity_num_layers: None, imagination_horizon: None, use_imagination_augmented: None, use_memory: false, memory_sequence_length: Some(64), memory_size: Some(256), time_horizon: 1000 } }
    pub fn ppo_ce() -> Self { Self { trainer_type: "ppo_ce".into(), batch_size: 2048, buffer_size: 20480, buffer_init_steps: 0, tau: 0.95, steps_per_update: 1.0, save_replay_buffer: false, init_entcoef: None, beta: Some(0.005), epsilon: Some(0.2), lambd: Some(0.95), num_epoch: Some(3), policy_delay: None, n_quantiles: None, n_to_drop: None, destructive_threshold: None, image_pad: None, entropy_temperature: None, adaptive_entropy_temperature: None, curiosity_strength: Some(0.01), curiosity_gamma: Some(0.99), curiosity_learning_rate: Some(0.0001), curiosity_hidden_units: Some(256), curiosity_num_layers: Some(2), imagination_horizon: Some(5), use_imagination_augmented: Some(true), use_memory: false, memory_sequence_length: Some(64), memory_size: Some(256), time_horizon: 1000 } }
    pub fn poca() -> Self { Self { trainer_type: "poca".into(), batch_size: 2048, buffer_size: 20480, buffer_init_steps: 0, tau: 0.0, steps_per_update: 1.0, save_replay_buffer: false, init_entcoef: Some(0.005), beta: Some(0.005), epsilon: Some(0.2), lambd: Some(0.95), num_epoch: Some(3), policy_delay: None, n_quantiles: None, n_to_drop: None, destructive_threshold: None, image_pad: None, entropy_temperature: None, adaptive_entropy_temperature: None, curiosity_strength: None, curiosity_gamma: None, curiosity_learning_rate: None, curiosity_hidden_units: None, curiosity_num_layers: None, imagination_horizon: None, use_imagination_augmented: None, use_memory: false, memory_sequence_length: Some(64), memory_size: Some(256), time_horizon: 1000 } }
}