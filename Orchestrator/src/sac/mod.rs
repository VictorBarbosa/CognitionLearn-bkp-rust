pub mod actor;
pub mod critic;
pub mod replay_buffer;

use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use crate::sac::actor::Actor;
use crate::sac::critic::Critic;
use crate::sac::replay_buffer::ReplayBuffer;
use crate::agent::RLAgent;
use std::collections::HashMap;

pub struct SAC {
    pub actor: Actor,
    pub critic1: Critic,
    pub critic2: Critic,
    pub target_critic1: Critic,
    pub target_critic2: Critic,
    
    pub vs: nn::VarStore,
    pub target_vs: nn::VarStore,
    pub actor_optimizer: nn::Optimizer,
    pub critic_optimizer: nn::Optimizer,
    pub alpha_optimizer: nn::Optimizer,
    
    pub log_alpha: Tensor,
    pub target_entropy: f64,
    
    pub replay_buffer: ReplayBuffer,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub gamma: f64,
    pub tau: f64,
    pub device: Device,
    pub train_count: usize,
    pub sensor_sizes: Option<Vec<i64>>,
    pub batch_size: usize,
    pub max_grad_norm: f64,
}

impl SAC {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden_dim: usize,
        buffer_capacity: usize,
        batch_size: usize,
        learning_rate: f64,
        gamma: f64,
        tau: f64,
        alpha_coef: f64,
        max_grad_norm: f64,
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
        
        // Target critics (independent varstore to avoid sharing weights)
        let mut target_vs = nn::VarStore::new(device);
        let target_root = target_vs.root();
        let target_critic1 = Critic::new(&(&target_root / "critic1"), obs_dim as i64, act_dim as i64, hidden_dim as i64);
        let target_critic2 = Critic::new(&(&target_root / "critic2"), obs_dim as i64, act_dim as i64, hidden_dim as i64);
        
        // Initial soft update (copy weights)
        target_vs.copy(&vs).expect("Failed to sync target critics");

        let log_alpha = root.var("log_alpha", &[1], nn::Init::Const(alpha_coef.ln())); // Trainable parameter initialized to log(alpha)
        let target_entropy = -(act_dim as f64);
        
        let actor_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build actor opt");
        let critic_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build critic opt");
        let alpha_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build alpha opt");

        Self {
            actor,
            critic1,
            critic2,
            target_critic1,
            target_critic2,
            vs,
            target_vs,
            actor_optimizer,
            critic_optimizer,
            alpha_optimizer,
            log_alpha,
            target_entropy,
            replay_buffer: ReplayBuffer::new(buffer_capacity, obs_dim, act_dim, device),
            obs_dim,
            act_dim,
            gamma,
            tau,
            device,
            train_count: 0,
            sensor_sizes,
            batch_size,
            max_grad_norm,
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

impl RLAgent for SAC {
    fn record_transition(&mut self, _agent_id: i32, obs: Vec<f32>, act: Vec<f32>, reward: f32, next_obs: Vec<f32>, done: bool) {
        self.replay_buffer.push(&obs, &act, reward, &next_obs, done);
    }

    fn train(&mut self) -> Option<HashMap<String, f32>> {
        if self.replay_buffer.len() < self.batch_size {
            return None;
        }

        let batch = self.replay_buffer.sample(self.batch_size);
        let alpha = self.log_alpha.exp();

        // 1. Update Critics
        let q_target = tch::no_grad(|| {
            let (next_actions, next_log_probs) = self.actor.sample(&batch.next_obs);
            let q1_target = self.target_critic1.forward(&batch.next_obs, &next_actions);
            let q2_target = self.target_critic2.forward(&batch.next_obs, &next_actions);
            let min_q_target = q1_target.min_other(&q2_target);
            
            &batch.rewards + &batch.done * self.gamma * (&min_q_target - &alpha * &next_log_probs)
        });
        
        let q1 = self.critic1.forward(&batch.obs, &batch.actions);
        let q2 = self.critic2.forward(&batch.obs, &batch.actions);
        
        let critic1_loss = (q1 - &q_target).pow_tensor_scalar(2.0).mean(Kind::Float);
        let critic2_loss = (q2 - &q_target).pow_tensor_scalar(2.0).mean(Kind::Float);
        let critic_loss = &critic1_loss + &critic2_loss;

        self.critic_optimizer.backward_step_clip(&critic_loss, self.max_grad_norm);

        // 2. Update Actor
        let (actions_new, log_probs_new) = self.actor.sample(&batch.obs);
        let q1_actor = self.critic1.forward(&batch.obs, &actions_new);
        let q2_actor = self.critic2.forward(&batch.obs, &actions_new);
        let min_q_actor = q1_actor.min_other(&q2_actor);
        
        let actor_loss = (&alpha * &log_probs_new - &min_q_actor).mean(Kind::Float);
        self.actor_optimizer.backward_step_clip(&actor_loss, self.max_grad_norm);

        // 3. Update Alpha (Temperature)
        let log_probs_detached = log_probs_new.detach();
        let alpha_loss = -(&self.log_alpha * (&log_probs_detached + Tensor::from(self.target_entropy).to_kind(Kind::Float).to_device(self.device))).mean(Kind::Float);
        self.alpha_optimizer.backward_step_clip(&alpha_loss, self.max_grad_norm);
        
        // Metrics
        let c_loss = critic_loss.double_value(&[]) as f32;
        let a_loss = actor_loss.double_value(&[]) as f32;
        let alpha_val = alpha.double_value(&[]) as f32;
        let ent_loss = alpha_loss.double_value(&[]) as f32;

        self.train_count += 1;
        
        // Soft update
        Self::soft_update(&mut self.target_vs, &self.vs, self.tau);
        
        let mut metrics = HashMap::new();
        metrics.insert("Losses/Value Loss".to_string(), c_loss / 2.0);
        metrics.insert("Losses/Policy Loss".to_string(), a_loss);
        metrics.insert("Policy/Entropy Coefficient".to_string(), alpha_val);
        metrics.insert("Losses/Entropy Loss".to_string(), ent_loss);
        Some(metrics)
    }

    fn select_action(&self, obs: &[f32], deterministic: bool) -> Vec<f32> {
        let obs_tensor = Tensor::from_slice(obs).to(self.device).reshape(&[1, self.obs_dim as i64]);
        
        let action = if deterministic {
            let (mean, _) = self.actor.forward(&obs_tensor);
            mean.tanh()
        } else {
            let (action, _) = self.actor.sample(&obs_tensor);
            action
        };
        
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
        // Access weights from Actor (refactored to expose l1, l2, mean)
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
