use tch::{nn, nn::OptimizerConfig, nn::ModuleT, Device, Tensor, Kind};
use crate::sac::actor::Actor;
use crate::sac::replay_buffer::ReplayBuffer;
use crate::agent::{RLAgent, ActionOutput};
use std::collections::HashMap;

pub struct CrossQCritic {
    pub net: nn::SequentialT,
}

impl CrossQCritic {
    pub fn new(p: &nn::Path, obs_dim: i64, act_dim: i64, hidden_dim: i64) -> Self {
        // CrossQ uses Batch Normalization in the critic to stabilize returns without a target network.
        // Architecture: Linear -> BN -> ReLU -> Linear -> BN -> ReLU -> Linear
        let net = nn::seq_t()
            .add(nn::linear(p / "l1", obs_dim + act_dim, hidden_dim, Default::default()))
            .add(nn::batch_norm1d(p / "bn1", hidden_dim, Default::default()))
            .add(nn::func_t(|xs, _train| xs.relu())) // func_t for activation if needed, or just func? relu is stateless.
            // Actually func(|xs| xs.relu()) implements ModuleT (it ignores train).
            // But SequentialT might expect ModuleT? Yes.
            // Let's check if func satisfies ModuleT. Usually yes.
            .add(nn::linear(p / "l2", hidden_dim, hidden_dim, Default::default()))
            .add(nn::batch_norm1d(p / "bn2", hidden_dim, Default::default()))
            .add(nn::func(|xs| xs.relu()))
            .add(nn::linear(p / "l3", hidden_dim, 1, Default::default()));
            
        Self { net }
    }

    pub fn forward_t(&self, obs: &Tensor, act: &Tensor, train: bool) -> Tensor {
        let x = Tensor::cat(&[obs, act], 1);
        self.net.forward_t(&x, train)
    }
}

pub struct CrossQ {
    pub actor: Actor,
    pub q1: CrossQCritic,
    pub q2: CrossQCritic,
    
    pub vs: nn::VarStore,
    
    pub actor_optimizer: nn::Optimizer,
    pub critic_optimizer: nn::Optimizer,
    pub alpha_optimizer: nn::Optimizer,
    
    pub log_alpha: Tensor,
    pub target_entropy: f64,
    
    pub replay_buffer: ReplayBuffer,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub gamma: f64,
    pub device: Device,
    pub train_count: usize,
    pub sensor_sizes: Option<Vec<i64>>,
    pub batch_size: usize,
    pub max_grad_norm: f64,
}

impl CrossQ {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden_dim: usize,
        buffer_capacity: usize,
        batch_size: usize,
        learning_rate: f64,
        gamma: f64,
        alpha_coef: f64,
        max_grad_norm: f64,
        device: Device,
        sensor_sizes: Option<Vec<i64>>,
        _memory_size: Option<usize>, // Placeholder
        _sequence_length: Option<usize>, // Placeholder
    ) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        // Actor (SAC style)
        let actor = Actor::new(&(&root / "actor"), obs_dim as i64, hidden_dim as i64, act_dim as i64);

        // Critics with Batch Norm
        let q1 = CrossQCritic::new(&(&root / "q1"), obs_dim as i64, act_dim as i64, hidden_dim as i64);
        let q2 = CrossQCritic::new(&(&root / "q2"), obs_dim as i64, act_dim as i64, hidden_dim as i64);
        
        let log_alpha = root.var("log_alpha", &[1], nn::Init::Const(alpha_coef.ln()));
        let target_entropy = -(act_dim as f64);
        
        let actor_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build actor opt");
        let critic_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build critic opt");
        let alpha_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build alpha opt");

        Self {
            actor,
            q1,
            q2,
            vs,
            actor_optimizer,
            critic_optimizer,
            alpha_optimizer,
            log_alpha,
            target_entropy,
            replay_buffer: ReplayBuffer::new(buffer_capacity, obs_dim, act_dim, device),
            obs_dim,
            act_dim,
            gamma,
            device,
            train_count: 0,
            sensor_sizes,
            batch_size,
            max_grad_norm,
        }
    }
}

impl RLAgent for CrossQ {
    fn record_transition(&mut self, _agent_id: i32, obs: Vec<f32>, act: ActionOutput, reward: f32, next_obs: Vec<f32>, done: bool) {
        self.replay_buffer.push(&obs, &act.continuous, reward, &next_obs, done);
    }

    fn train(&mut self) -> Option<HashMap<String, f32>> {
        if self.replay_buffer.len() < self.batch_size {
            return None;
        }

        self.train_count += 1;
        let batch = self.replay_buffer.sample(self.batch_size);
        let alpha = self.log_alpha.exp();

        // CrossQ Core Logic:
        // Concatenate current (s, a) and next (s', a') into a single batch for the critics.
        // This is crucial for Batch Normalization to work correctly on the mixed distribution.
        
        let (next_actions, next_log_probs) = tch::no_grad(|| {
            self.actor.sample(&batch.next_obs)
        });

        // 1. Prepare Cross Batch
        // [batch.obs, batch.next_obs]
        let cross_obs = Tensor::cat(&[&batch.obs, &batch.next_obs], 0); 
        // [batch.actions, next_actions]
        let cross_acts = Tensor::cat(&[&batch.actions, &next_actions], 0);

        // 2. Forward pass through critics (updates BN stats)
        let cross_q1 = self.q1.forward_t(&cross_obs, &cross_acts, true);
        let cross_q2 = self.q2.forward_t(&cross_obs, &cross_acts, true);

        // 3. Split outputs back
        // q(s, a) is the first half
        let q1_curr = cross_q1.narrow(0, 0, self.batch_size as i64);
        let q2_curr = cross_q2.narrow(0, 0, self.batch_size as i64);
        
        // q(s', a') is the second half
        let q1_next = cross_q1.narrow(0, self.batch_size as i64, self.batch_size as i64);
        let q2_next = cross_q2.narrow(0, self.batch_size as i64, self.batch_size as i64);

        // 4. Calculate Target
        // Note: In CrossQ, we use the *current* network outputs for the target, but we detach them.
        // Since we don't have a target network, this is the "target".
        let min_q_next = q1_next.min_other(&q2_next);
        let target_q = &batch.rewards + &batch.done * self.gamma * (min_q_next - &alpha * &next_log_probs).detach();

        // 5. Critic Loss
        let critic_loss = (q1_curr - &target_q).pow_tensor_scalar(2.0).mean(Kind::Float) + 
                          (q2_curr - &target_q).pow_tensor_scalar(2.0).mean(Kind::Float);
        
        self.critic_optimizer.backward_step_clip(&critic_loss, self.max_grad_norm);

        // 6. Actor Update (standard SAC style, using current Q)
        // We need to re-evaluate Q because we updated critics? 
        // Original CrossQ paper suggests iterating actor update *after* critic.
        // We can reuse the same batch.obs (first half of cross batch).
        // To be safe and precise, let's execute actor sample on obs again.
        
        let (actions_new, log_probs_new) = self.actor.sample(&batch.obs);
        let q1_pi = self.q1.forward_t(&batch.obs, &actions_new, true);
        let q2_pi = self.q2.forward_t(&batch.obs, &actions_new, true);
        let min_q_pi = q1_pi.min_other(&q2_pi);
        
        let actor_loss = (&alpha * &log_probs_new - min_q_pi).mean(Kind::Float);
        self.actor_optimizer.backward_step_clip(&actor_loss, self.max_grad_norm);

        // 7. Alpha Update
        let log_probs_detached = log_probs_new.detach();
        let alpha_loss = -(&self.log_alpha * (&log_probs_detached + Tensor::from(self.target_entropy).to_kind(Kind::Float).to_device(self.device))).mean(Kind::Float);
        self.alpha_optimizer.backward_step_clip(&alpha_loss, self.max_grad_norm);

        let c_loss = critic_loss.double_value(&[]) as f32;
        let a_loss = actor_loss.double_value(&[]) as f32;
        let alpha_val = alpha.double_value(&[]) as f32;
        let ent_loss = alpha_loss.double_value(&[]) as f32;

        let mut metrics = HashMap::new();
        metrics.insert("Losses/Value Loss".to_string(), c_loss / 2.0);
        metrics.insert("Losses/Policy Loss".to_string(), a_loss);
        metrics.insert("Policy/Entropy Coefficient".to_string(), alpha_val);
        metrics.insert("Losses/Entropy Loss".to_string(), ent_loss);

        Some(metrics)
    }

    fn select_action(&self, obs: &[f32], deterministic: bool) -> ActionOutput {
        let obs_tensor = Tensor::from_slice(obs).to(self.device).reshape(&[1, self.obs_dim as i64]);
        
        let action = if deterministic {
            let (mean, _) = self.actor.forward(&obs_tensor);
            mean.tanh()
        } else {
            let (action, _) = self.actor.sample(&obs_tensor);
            action
        };
        
        let action_flat: Vec<f32> = action.flatten(0, -1).try_into().unwrap();
        ActionOutput {
            continuous: action_flat,
            discrete: vec![]
        }
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
