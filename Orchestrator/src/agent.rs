use std::collections::HashMap;


use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AgentType {
    SAC,
    PPO,
    PPO_ET, // Added
    PPO_CE, // Added
    TD3,
    BC,
    TDSAC,
    TQC,
    CrossQ,
    DRQV2,
}

pub trait RLAgent: Send {
    fn record_transition(
        &mut self,
        agent_id: i32,
        obs: Vec<f32>,
        act: Vec<f32>,
        reward: f32,
        next_obs: Vec<f32>,
        done: bool,
    );

    /// Performs a training step and returns metrics for logging
    fn train(&mut self) -> Option<HashMap<String, f32>>;

    fn select_action(&self, obs: &[f32], deterministic: bool) -> Vec<f32>;

    fn get_obs_dim(&self) -> usize;
    fn get_act_dim(&self) -> usize;
    fn get_buffer_size(&self) -> usize;
    fn get_training_threshold(&self) -> usize;

    fn save(&self, path: &str) -> std::io::Result<()>;
    fn load(&mut self, path: &str) -> std::io::Result<()>;
    fn export_onnx(&self, _path: &str) -> std::io::Result<()> {
        Ok(()) // Default implementation does nothing
    }
}
