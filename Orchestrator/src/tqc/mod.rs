use tch::{nn, nn::OptimizerConfig, nn::Module, Device, Tensor, Kind};
use crate::sac::actor::Actor;
use crate::sac::replay_buffer::{ReplayBuffer, Transition};
use crate::agent::RLAgent;
use std::collections::HashMap;

pub struct QuantileCritic {
    pub net: nn::Sequential,
}

impl QuantileCritic {
    pub fn new(p: &nn::Path, obs_dim: i64, act_dim: i64, hidden_dim: i64, n_quantiles: i64) -> Self {
        let net = nn::seq()
            .add(nn::linear(p / "l1", obs_dim + act_dim, hidden_dim, Default::default()))
            .add(nn::func(|xs| xs.relu()))
            .add(nn::linear(p / "l2", hidden_dim, hidden_dim, Default::default()))
            .add(nn::func(|xs| xs.relu()))
            .add(nn::linear(p / "l3", hidden_dim, n_quantiles, Default::default()));
            
        Self { net }
    }

    pub fn forward(&self, obs: &Tensor, act: &Tensor) -> Tensor {
        let x = Tensor::cat(&[obs, act], 1);
        self.net.forward(&x)
    }
}

pub struct TQC {
    pub actor: Actor,
    pub critics: Vec<QuantileCritic>,
    pub target_critics: Vec<QuantileCritic>,
    
    pub vs: nn::VarStore,
    pub target_vs: nn::VarStore,
    
    pub actor_optimizer: nn::Optimizer,
    pub critic_optimizer: nn::Optimizer, // Optimizes all critics together (or list of optimizers?) 
    // We'll use one optimizer for all critics if they are in same VarStore, but wait, usually we want strict separation?
    // In tch, if we put them all in `vs / "critics"`, one optimizer is fine.
    
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

    // TQC Specifics
    pub top_quantiles_to_drop: i64,
    pub n_quantiles: i64,
    pub n_nets: usize,
}

impl TQC {
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
        n_quantiles: usize,
        n_to_drop: usize,
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

        // Critics
        let n_nets = 5;
        let n_quantiles = n_quantiles as i64;
        let top_quantiles_to_drop_per_net = n_to_drop as i64;
        let top_quantiles_to_drop = top_quantiles_to_drop_per_net * n_nets as i64;
        
        let mut critics = Vec::new();
        for i in 0..n_nets {
            critics.push(QuantileCritic::new(&(&root / format!("critic_{}", i)), obs_dim as i64, act_dim as i64, hidden_dim as i64, n_quantiles));
        }

        // Targets
        let mut target_vs = nn::VarStore::new(device);
        let target_root = target_vs.root();
        
        let mut target_critics = Vec::new();
        for i in 0..n_nets {
            target_critics.push(QuantileCritic::new(&(&target_root / format!("critic_{}", i)), obs_dim as i64, act_dim as i64, hidden_dim as i64, n_quantiles));
        }
        
        target_vs.copy(&vs).expect("Failed to sync target critics");

        let log_alpha = root.var("log_alpha", &[1], nn::Init::Const(alpha_coef.ln()));
        let target_entropy = -(act_dim as f64);
        
        let actor_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build actor opt");
        let critic_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build critic opt");
        let alpha_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build alpha opt");

        Self {
            actor,
            critics,
            target_critics,
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
            n_quantiles,
            n_nets,
            top_quantiles_to_drop,
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

    fn quantile_huber_loss(current_quantiles: &Tensor, target_quantiles: &Tensor) -> Tensor {
        // current: [batch, n_nets, n_quantiles] -> expanded to [batch, n_nets, n_quantiles, 1]
        // target: [batch, total_remaining_quantiles] -> expanded to [batch, 1, 1, total_remaining]
        // This is expensive O(N*M).
        // Let's allow broadcasting.
        
        // Pytorch impl usually does:
        // diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)  (Shape: batch, n_quantiles_student, n_quantiles_teacher)
        // Here current has structure structure.
        
        // Let's follow the standard implementation logic:
        // current_quantiles shape: [batch, n_nets * n_quantiles] 
        // target_quantiles shape: [batch, target_n_quantiles]
        
        let n_quantiles = current_quantiles.size()[1];
        // Unused intermediate variables removed/commented
        // let n_target_quantiles = target_quantiles.size()[1];
        // let batch_size = current_quantiles.size()[0];

        let current = current_quantiles.unsqueeze(2); // [B, N, 1]
        let target = target_quantiles.unsqueeze(1).detach(); // [B, 1, M]
        
        let diff = &target - &current; // [B, N, M]
        
        // Manual Huber Loss Logic
        // let huber_loss = (diff.abs() - 1.0).relu().neg() + diff.pow_tensor_scalar(2.0) * 0.5 + diff.abs();

        // Or using smooth_l1_loss logic
       
        // Quantile regression logic:
        // loss = inclusive_prob * huber
        // inclusive_prob = abs(tau - (diff < 0).float())
        
        // Construct taus (cumulative probabilities)
        // taus = (torch.arange(n_quantiles) + 0.5) / n_quantiles
        let device = current_quantiles.device();
        let taus = (Tensor::arange(n_quantiles, (Kind::Float, device)) + 0.5) / (n_quantiles as f64);
        let taus = taus.view([1, n_quantiles, 1]); // Broadcast ready
        
        // Unused element_wise_loss block removed
        /*
        let element_wise_loss = if diff.abs().less_equal(1.0).all().int64_value(&[]) == 1 {
            diff.pow_tensor_scalar(2.0) * 0.5 
        } else {
             diff.abs() - 0.5 
        };
        */
        
        // Re-implement huber properly elementwise
        let cond = diff.abs().less(1.0);
        let huber = cond.type_as(&diff) * 0.5 * diff.pow_tensor_scalar(2.0) + cond.logical_not().type_as(&diff) * (diff.abs() - 0.5);
        
        let delta = (diff.less(0.0).type_as(&diff) - taus).abs().detach();
        
        (delta * huber).mean(Kind::Float)
    }
}

impl RLAgent for TQC {
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
        if self.replay_buffer.len() < self.batch_size {
            return None;
        }

        self.train_count += 1;
        let batch = self.replay_buffer.sample(self.batch_size);
        let alpha = self.log_alpha.exp();

        // 1. Target calculation
        let target_quantiles = tch::no_grad(|| {
            let (next_actions, next_log_probs) = self.actor.sample(&batch.next_obs);
            
            // Collect quantiles from all target critics
            let mut quantiles_list = Vec::new();
            for critic in &self.target_critics {
                quantiles_list.push(critic.forward(&batch.next_obs, &next_actions)); // [batch, n_quantiles]
            }
            let next_target_quantiles = Tensor::cat(&quantiles_list, 1); // [batch, n_nets * n_quantiles]
            
            // Sort
            let (sorted_quantiles, _) = next_target_quantiles.sort(1, false);
            
            // Truncate (Remove top k)
            let total_quantiles = self.n_nets as i64 * self.n_quantiles;
            let n_target_quantiles = total_quantiles - self.top_quantiles_to_drop;
            let truncated_quantiles = sorted_quantiles.narrow(1, 0, n_target_quantiles);
            
            // Apply Bellman update: r + gamma * (q - alpha * log_pi)
            &batch.rewards + &batch.done * self.gamma * (truncated_quantiles - &alpha * next_log_probs.reshape([self.batch_size as i64, 1]))
        });

        // 2. Critic Update
        let mut current_quantiles_list = Vec::new();
        for critic in &self.critics {
             current_quantiles_list.push(critic.forward(&batch.obs, &batch.actions));
        }
        let current_quantiles = Tensor::cat(&current_quantiles_list, 1); // [B, TotalQ]
        
        // This is a simplified loss calculation. 
        // Standard TQC sums loss over each critic independently against the target.
        // But since we concatenated them, we can try to compute it in bulk IF we interpret appropriately.
        // Actually, TQC implementation usually does loop over critics.
        // Let's replicate standard TQC loss: Sum of (QuantileHuberLoss(critic_i, target))
        
        // But for efficiency, we can treat [B, n_nets*n_quantiles] vs [B, n_target_quantiles]
        let critic_loss = Self::quantile_huber_loss(&current_quantiles, &target_quantiles);
        
        self.critic_optimizer.backward_step_clip(&critic_loss, self.max_grad_norm);

        // 3. Actor Update
        let (actions_new, log_probs_new) = self.actor.sample(&batch.obs);

        // Calculate Q for actor: Mean of truncated quantiles from current critics
        let mut actor_q_list = Vec::new();
         for critic in &self.critics {
             actor_q_list.push(critic.forward(&batch.obs, &actions_new));
        }
        let actor_q_concat = Tensor::cat(&actor_q_list, 1);
        let (actor_q_sorted, _) = actor_q_concat.sort(1, false);
        let total_quantiles = self.n_nets as i64 * self.n_quantiles;
        let n_target_quantiles = total_quantiles - self.top_quantiles_to_drop;
        let actor_q_truncated = actor_q_sorted.narrow(1, 0, n_target_quantiles);
        let q_mean = actor_q_truncated.mean_dim(&[1i64][..], true, Kind::Float); // [B, 1]

        let actor_loss = (&alpha * &log_probs_new - q_mean).mean(Kind::Float);
        self.actor_optimizer.backward_step_clip(&actor_loss, self.max_grad_norm);

        // 4. Alpha Update
        let log_probs_detached = log_probs_new.detach();
        let alpha_loss = -(&self.log_alpha * (&log_probs_detached + Tensor::from(self.target_entropy).to_kind(Kind::Float).to_device(self.device))).mean(Kind::Float);
        self.alpha_optimizer.backward_step_clip(&alpha_loss, self.max_grad_norm);
        
        // Soft Updates
        Self::soft_update(&mut self.target_vs, &self.vs, self.tau);

        let c_loss = critic_loss.double_value(&[]) as f32;
        let a_loss = actor_loss.double_value(&[]) as f32;
        let alpha_val = alpha.double_value(&[]) as f32;
        let ent_loss = alpha_loss.double_value(&[]) as f32;

        let mut metrics = HashMap::new();
        metrics.insert("Losses/Value Loss".to_string(), c_loss);
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
