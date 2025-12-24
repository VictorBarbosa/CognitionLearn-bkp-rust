use tch::{Device, Tensor};

#[derive(Debug, Clone)]
pub struct PPOTransition {
    pub agent_id: i32, // Added
    pub observation: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub next_observation: Vec<f32>,
    pub log_prob: f32,
    pub done: bool,
    pub value: f32,
}

pub struct Rollout {
    // Store transitions grouped by agent to maintain sequence integrity
    pub agent_transitions: std::collections::HashMap<i32, Vec<PPOTransition>>,
    pub total_count: usize,
}

impl Rollout {
    pub fn new() -> Self {
        Self { 
            agent_transitions: std::collections::HashMap::new(),
            total_count: 0,
        }
    }

    pub fn add(&mut self, t: PPOTransition) {
        self.total_count += 1;
        self.agent_transitions
            .entry(t.agent_id)
            .or_insert_with(Vec::new)
            .push(t);
    }

    pub fn clear(&mut self) {
        self.agent_transitions.clear();
        self.total_count = 0;
    }

    pub fn len(&self) -> usize {
        self.total_count
    }

    // Helper to get all agent IDs currently in buffer
    pub fn agent_ids(&self) -> Vec<i32> {
        self.agent_transitions.keys().cloned().collect()
    }
}
