pub mod actor;
pub mod critic;

use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use crate::td3::actor::Actor;
use crate::td3::critic::Critic;
use crate::sac::replay_buffer::{ReplayBuffer, Transition};
use crate::agent::RLAgent;
use std::collections::HashMap;

pub struct TD3 {
    pub actor: Actor,
    pub target_actor: Actor,
    pub critic1: Critic,
    pub target_critic1: Critic,
    pub critic2: Critic,
    pub target_critic2: Critic,

    pub vs: nn::VarStore,
    pub target_vs: nn::VarStore,
    pub actor_optimizer: nn::Optimizer,
    pub critic_optimizer: nn::Optimizer,

    pub replay_buffer: ReplayBuffer,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub gamma: f64,
    pub tau: f64,
    pub policy_noise: f64,
    pub noise_clip: f64,
    pub policy_freq: usize,
    pub exploration_noise: f64,
    pub device: Device,
    pub train_count: usize,
    pub sensor_sizes: Option<Vec<i64>>,
}

impl TD3 {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden_dim: usize,
        buffer_capacity: usize,
        learning_rate: f64,
        gamma: f64,
        tau: f64,
        policy_freq: usize,
        device: Device,
        sensor_sizes: Option<Vec<i64>>,
        _memory_size: Option<usize>, // Placeholder
        _sequence_length: Option<usize>, // Placeholder
    ) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let actor = Actor::new(&(&root / "actor"), obs_dim as i64, hidden_dim as i64, act_dim as i64);
        let critic1 = Critic::new(&(&root / "critic1"), obs_dim as i64, act_dim as i64, hidden_dim as i64);
        let critic2 = Critic::new(&(&root / "critic2"), obs_dim as i64, act_dim as i64, hidden_dim as i64);

        // Target networks (independent varstore)
        let mut target_vs = nn::VarStore::new(device);
        let target_root = target_vs.root();
        let target_actor = Actor::new(&(&target_root / "actor"), obs_dim as i64, hidden_dim as i64, act_dim as i64);
        let target_critic1 = Critic::new(&(&target_root / "critic1"), obs_dim as i64, act_dim as i64, hidden_dim as i64);
        let target_critic2 = Critic::new(&(&target_root / "critic2"), obs_dim as i64, act_dim as i64, hidden_dim as i64);

        // Initial soft update (copy weights)
        target_vs.copy(&vs).expect("Failed to sync target networks");

        let actor_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build actor opt");
        let critic_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build critic opt");

        Self {
            actor,
            target_actor,
            critic1,
            target_critic1,
            critic2,
            target_critic2,
            vs,
            target_vs,
            actor_optimizer,
            critic_optimizer,
            replay_buffer: ReplayBuffer::new(buffer_capacity, obs_dim, act_dim, device),
            obs_dim,
            act_dim,
            gamma,
            tau,
            policy_noise: 0.2,
            noise_clip: 0.5,
            policy_freq,
            exploration_noise: 0.1,
            device,
            train_count: 0,
            sensor_sizes,
        }
    }

    fn soft_update(target: &mut nn::VarStore, source: &nn::VarStore, tau: f64) {
        tch::no_grad(|| {
            for (name, mut target_var) in target.variables() {
                if let Some(source_var) = source.variables().get(&name) {
                    let new_val = target_var.shallow_clone() * (1.0 - tau) + source_var.shallow_clone() * tau;
                    target_var.set_data(&new_val);
                }
            }
        });
    }
}

impl RLAgent for TD3 {
    fn record_transition(&mut self, _agent_id: i32, obs: Vec<f32>, act: Vec<f32>, reward: f32, next_obs: Vec<f32>, done: bool) {
        self.replay_buffer.push(Transition {
            obs,
            actions: act,
            reward,
            next_obs,
            done,
        });
    }

    fn train(&mut self) -> Option<HashMap<String, f32>> {
        let batch_size = 256;
        if self.replay_buffer.len() < batch_size {
            return None;
        }

        self.train_count += 1;
        let batch = self.replay_buffer.sample(batch_size);

        // 1. Update Critics
        let q_target = tch::no_grad(|| {
            // Target Policy Smoothing
            let next_actions = self.target_actor.forward(&batch.next_obs);
            let noise = Tensor::randn_like(&next_actions) * self.policy_noise;
            let noise = noise.clamp(-self.noise_clip, self.noise_clip);
            let next_actions = (next_actions + noise).clamp(-1.0, 1.0);

            let target_q1 = self.target_critic1.forward(&batch.next_obs, &next_actions);
            let target_q2 = self.target_critic2.forward(&batch.next_obs, &next_actions);
            let target_q = target_q1.min_other(&target_q2);

            &batch.rewards + &batch.done * self.gamma * target_q
        });

        let q1 = self.critic1.forward(&batch.obs, &batch.actions);
        let q2 = self.critic2.forward(&batch.obs, &batch.actions);

        let critic_loss = (q1 - &q_target).pow_tensor_scalar(2.0).mean(Kind::Float) + 
                          (q2 - &q_target).pow_tensor_scalar(2.0).mean(Kind::Float);

        self.critic_optimizer.backward_step(&critic_loss);

        let c_loss = critic_loss.double_value(&[]) as f32;
        let mut metrics = HashMap::new();
        metrics.insert("Losses/Value Loss".to_string(), c_loss / 2.0);

        // 2. Delayed Policy Updates
        if self.train_count % self.policy_freq == 0 {
            let actions_new = self.actor.forward(&batch.obs);
            let actor_loss = -self.critic1.forward(&batch.obs, &actions_new).mean(Kind::Float);
            
            self.actor_optimizer.backward_step(&actor_loss);

            // Soft Update
            Self::soft_update(&mut self.target_vs, &self.vs, self.tau);

            metrics.insert("Losses/Policy Loss".to_string(), actor_loss.double_value(&[]) as f32);
        }

        Some(metrics)
    }

    fn select_action(&self, obs: &[f32], deterministic: bool) -> Vec<f32> {
        let obs_tensor = Tensor::from_slice(obs).to(self.device).reshape(&[1, self.obs_dim as i64]);
        
        // No grad for inference
        let action = tch::no_grad(|| {
            let action = self.actor.forward(&obs_tensor);
            if !deterministic {
                let noise = Tensor::randn_like(&action) * self.exploration_noise;
                (action + noise).clamp(-1.0, 1.0)
            } else {
                action
            }
        });
        
        let action_flat = action.flatten(0, -1);
        action_flat.try_into().unwrap()
    }

    fn get_obs_dim(&self) -> usize { self.obs_dim }
    fn get_act_dim(&self) -> usize { self.act_dim }
    fn get_buffer_size(&self) -> usize { self.replay_buffer.len() }
    fn get_training_threshold(&self) -> usize { 256 }

    fn save(&self, path: &str) -> std::io::Result<()> {
        self.vs.save(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    fn load(&mut self, path: &str) -> std::io::Result<()> {
        self.vs.load(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    fn export_onnx(&self, path: &str) -> std::io::Result<()> {
        // Access weights from Actor (exposed l1, l2, l3)
        let l1_w = &self.actor.l1.ws;
        let l1_b = self.actor.l1.bs.as_ref().expect("L1 bias missing");
        let l2_w = &self.actor.l2.ws;
        let l2_b = self.actor.l2.bs.as_ref().expect("L2 bias missing");
        let l3_w = &self.actor.l3.ws;
        let l3_b = self.actor.l3.bs.as_ref().expect("L3 bias missing");

        crate::onnx_utils::export_td3_onnx(
            l1_w, l1_b,
            l2_w, l2_b,
            l3_w, l3_b,
            self.obs_dim as i64,
            self.act_dim as i64,
            path,
            &self.sensor_sizes
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}
