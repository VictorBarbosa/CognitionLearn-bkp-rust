use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AgentType {
    SAC,
    PPO,
    PPO_ET,
    PPO_CE,
    TD3,
    BC,
    TDSAC,
    TQC,
    CrossQ,
    DRQV2,
}

#[derive(Debug, Clone, Default)]
pub struct ActionOutput {
    pub continuous: Vec<f32>,
    pub discrete: Vec<i32>,
}

pub trait RLAgent: Send {
    fn record_transition(
        &mut self,
        agent_id: i32,
        obs: Vec<f32>,
        act: ActionOutput,
        reward: f32,
        next_obs: Vec<f32>,
        done: bool,
    );

    /// Performs a training step and returns metrics for logging
    fn train(&mut self) -> Option<HashMap<String, f32>>;

    fn select_action(&self, obs: &[f32], deterministic: bool) -> ActionOutput;

    fn get_obs_dim(&self) -> usize;
    fn get_act_dim(&self) -> usize; // Total flattened dimension (for continuous) or placeholder?
    // We might need explicit getters for structure, but for now we'll stick to this and cast inside if needed.
    
    fn get_buffer_size(&self) -> usize;
    fn get_training_threshold(&self) -> usize;

    fn save(&self, path: &str) -> std::io::Result<()>;
    fn load(&mut self, path: &str) -> std::io::Result<()>;
    fn export_onnx(&self, _path: &str) -> std::io::Result<()> {
        Ok(()) // Default implementation does nothing
    }
}
