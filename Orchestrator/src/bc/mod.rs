use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use crate::sac::actor::Actor;
use crate::sac::replay_buffer::ReplayBuffer;
use crate::agent::RLAgent;
use std::collections::HashMap;

pub struct BC {
    pub actor: Actor,
    pub optimizer: nn::Optimizer,
    pub vs: nn::VarStore,
    pub replay_buffer: ReplayBuffer,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub device: Device,
    pub train_count: usize,
    pub sensor_sizes: Option<Vec<i64>>,
    
    // BC Hyperparameters
    pub batch_size: usize,
}

impl BC {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden_dim: usize,
        buffer_capacity: usize,
        batch_size: usize,
        learning_rate: f64,
        device: Device,
        sensor_sizes: Option<Vec<i64>>,
        _memory_size: Option<usize>, // Placeholder
        _sequence_length: Option<usize>, // Placeholder
    ) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        // Actor (SAC style)
        let actor = Actor::new(&(&root / "actor"), obs_dim as i64, hidden_dim as i64, act_dim as i64);

        let optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build actor opt");

        Self {
            actor,
            optimizer,
            vs,
            replay_buffer: ReplayBuffer::new(buffer_capacity, obs_dim, act_dim, device),
            obs_dim,
            act_dim,
            device,
            train_count: 0,
            sensor_sizes,
            batch_size,
        }
    }
}

impl RLAgent for BC {
    fn record_transition(&mut self, _agent_id: i32, obs: Vec<f32>, act: Vec<f32>, reward: f32, next_obs: Vec<f32>, done: bool) {
        self.replay_buffer.push(&obs, &act, reward, &next_obs, done);
    }

    fn train(&mut self) -> Option<HashMap<String, f32>> {
        if self.replay_buffer.len() < self.batch_size {
            return None;
        }

        self.train_count += 1;
        let batch = self.replay_buffer.sample(self.batch_size);

        // BC Loss: MSE(Actor(s).mean, a_expert)
        // We use the 'mean' output from the SAC actor (which outputs mean + std)
        let (mean, _) = self.actor.forward(&batch.obs);
        
        // Use tanh for strict action bounding [-1, 1], matching SAC output
        let pred_action = mean.tanh();
        
        let mse_loss = (pred_action - &batch.actions).pow_tensor_scalar(2.0).mean(Kind::Float);

        self.optimizer.backward_step(&mse_loss);

        let loss_val = mse_loss.double_value(&[]) as f32;

        let mut metrics = HashMap::new();
        metrics.insert("Losses/BC Loss".to_string(), loss_val);

        Some(metrics)
    }

    fn select_action(&self, obs: &[f32], deterministic: bool) -> Vec<f32> {
        let obs_tensor = Tensor::from_slice(obs).to(self.device).reshape(&[1, self.obs_dim as i64]);
        
        let action = if deterministic {
            let (mean, _) = self.actor.forward(&obs_tensor);
            mean.tanh()
        } else {
            // Even in BC, we might want to sample if we want exploration? 
            // Standard BC is deterministic deployment.
            // But if reusing SAC actor, we can sample. 
            // For BC, typically we just want the mean.
            let (mean, _) = self.actor.forward(&obs_tensor);
            mean.tanh()
        };
        
        let action_flat = action.flatten(0, -1);
        action_flat.try_into().unwrap()
    }

    fn get_obs_dim(&self) -> usize { self.obs_dim }
    fn get_act_dim(&self) -> usize { self.act_dim }
    fn get_buffer_size(&self) -> usize { self.replay_buffer.len() }
    fn get_training_threshold(&self) -> usize { self.batch_size }

    fn save(&self, path: &str) -> std::io::Result<()> {
        self.vs.save(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    fn load(&mut self, path: &str) -> std::io::Result<()> {
        self.vs.load(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
    
    fn export_onnx(&self, path: &str) -> std::io::Result<()> {
        // Reuse SAC export (structure is identical)
        let l1_w = &self.actor.l1.ws;
        let l1_b = self.actor.l1.bs.as_ref().expect("L1 bias missing");
        let l2_w = &self.actor.l2.ws;
        let l2_b = self.actor.l2.bs.as_ref().expect("L2 bias missing");
        let mean_w = &self.actor.mean.ws;
        let mean_b = self.actor.mean.bs.as_ref().expect("Mean bias missing");

        crate::onnx_utils::export_sac_onnx(
            l1_w, l1_b,
            l2_w, l2_b,
            mean_w, mean_b,
            self.obs_dim as i64,
            self.act_dim as i64,
            path,
            &self.sensor_sizes
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}
