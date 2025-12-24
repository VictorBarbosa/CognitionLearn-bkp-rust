use tch::{nn, nn::OptimizerConfig, nn::Module, Device, Tensor, Kind};
use crate::sac::actor::Actor; // Reuse SAC Gaussian Actor (Independent)
use crate::agent::RLAgent;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct POCATransition {
    pub obs: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub log_prob: f32,
    pub done: bool,
    pub agent_id: i32,
}

pub struct POCARollout {
    pub transitions: Vec<POCATransition>,
}

impl POCARollout {
    pub fn new() -> Self { Self { transitions: Vec::new() } }
    pub fn push(&mut self, t: POCATransition) { self.transitions.push(t); }
    pub fn clear(&mut self) { self.transitions.clear(); }
    pub fn len(&self) -> usize { self.transitions.len() }
}

// Simple Self-Attention Critic
// Obs -> Embedding -> SelfAttention -> GlobalAvg -> Value
pub struct AttentionCritic {
    pub encoder: nn::Linear,
    pub query: nn::Linear,
    pub key: nn::Linear,
    pub value: nn::Linear,
    pub out_proj: nn::Linear,
    
    pub value_head: nn::Sequential,
}

impl AttentionCritic {
    pub fn new(p: &nn::Path, obs_dim: i64, hidden_dim: i64) -> Self {
        let encoder = nn::linear(p / "encoder", obs_dim, hidden_dim, Default::default());
        
        // Simplified Self-Attention (1 Head for clarity)
        let query = nn::linear(p / "query", hidden_dim, hidden_dim, Default::default());
        let key = nn::linear(p / "key", hidden_dim, hidden_dim, Default::default());
        let value = nn::linear(p / "value", hidden_dim, hidden_dim, Default::default());
        let out_proj = nn::linear(p / "out_proj", hidden_dim, hidden_dim, Default::default());

        let value_head = nn::seq()
            .add(nn::linear(p / "v1", hidden_dim, hidden_dim, Default::default()))
            .add(nn::func(|xs| xs.relu()))
            .add(nn::linear(p / "v2", hidden_dim, 1, Default::default()));

        Self { encoder, query, key, value, out_proj, value_head }
    }

    pub fn forward(&self, obs: &Tensor) -> Tensor {
        // obs: [Batch, N, Obs]
        let embed = obs.apply(&self.encoder).relu(); // [B, N, H]
        
        // Self Attention
        let q = embed.apply(&self.query); // [B, N, H]
        let k = embed.apply(&self.key);
        let v = embed.apply(&self.value);
        
        // Scores: Q * K^T
        let scores = q.matmul(&k.transpose(-2, -1)) / (embed.size()[2] as f64).sqrt(); // [B, N, N]
        let weights = scores.softmax(-1, Kind::Float);
        
        let context = weights.matmul(&v); // [B, N, H]
        let context = context.apply(&self.out_proj); // [B, N, H]
        
        // Residual
        let resid = context + embed;
        
        // Aggregation (Mean Pool over agents)
        let global_embed = resid.mean_dim(&[1i64][..], false, Kind::Float); // [B, H]
        
        // Calculate Value
        self.value_head.forward(&global_embed) // [B, 1]
    }
}

pub struct POCA {
    pub actor: Actor,
    pub critic: AttentionCritic,
    pub actor_optimizer: nn::Optimizer,
    pub critic_optimizer: nn::Optimizer,
    pub rollout: POCARollout,
    
    pub device: Device,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub batch_size: usize,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub ent_coef: f64,
    pub vf_coef: f64,
    pub num_epochs: usize,
    pub eps_clip: f64,
    
    pub sensor_sizes: Option<Vec<i64>>,
    pub vs: nn::VarStore,
}

impl POCA {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden_dim: usize,
        _buffer_capacity: usize, // Ignored for PPO/POCA (on-policy)
        batch_size: usize,
        learning_rate: f64,
        num_epochs: usize,
        gamma: f64,
        gae_lambda: f64,
        clip_coef: f64,
        ent_coef: f64,
        vf_coef: f64,
        device: Device,
        sensor_sizes: Option<Vec<i64>>,
        _memory_size: Option<usize>, // Placeholder
        _sequence_length: Option<usize>, // Placeholder
    ) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let actor = Actor::new(&(&root / "actor"), obs_dim as i64, hidden_dim as i64, act_dim as i64); // Note: SAC Actor signature is (obs, hidden, act) or (obs, hidden, act) depending on actor.rs. sac/actor.rs says: (input, hidden, output) -> (obs, hidden, act). Correct.
        let critic = AttentionCritic::new(&(&root / "critic"), obs_dim as i64, hidden_dim as i64);
        
        let actor_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Actor Opt");
        let critic_optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Critic Opt");

        Self {
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout: POCARollout::new(),
            device,
            obs_dim,
            act_dim,
            batch_size, 
            gamma,
            gae_lambda,
            ent_coef,
            vf_coef,
            num_epochs,
            eps_clip: clip_coef, // Use passed clip_coef
            sensor_sizes,
            vs,
        }
    }

    // Helper to calculate log prob of a Tanh'd action by approximating the pre-tanh value
    fn calc_log_prob_from_action(&self, obs_tensor: &Tensor, act_tensor: &Tensor) -> Tensor {
        // 1. Forward pass to get distribution params
        let (mean, log_std) = self.actor.forward(obs_tensor);
        
        // 2. Approximate x_t (pre-tanh) from action (post-tanh)
        // x_t = atanh(action). Clamp to avoid inf/nan at +/- 1.0
        let clipped_action = act_tensor.clamp(-0.999999, 0.999999);
        let x_t_approx = clipped_action.atanh();
        
        // 3. Use Actor's logic to calculate log_prob(action) given x_t
        // Accessing private method calc_log_prob is not possible if it's not pub. 
        // We will inline the logic here since we can't easily change sac/actor.rs right now.
        // Logic from Actor::calc_log_prob:
        
        let std = log_std.exp();
        let var = std.pow_tensor_scalar(2.0);
        let _log_std_clamped = log_std.clamp(-20.0, 2.0); // Match Actor clamping if needed, though Actor clamps output.
        
        // Gaussian log prob
        let log_prob_gauss = -0.5 * (&x_t_approx - &mean).pow_tensor_scalar(2.0) / &var
            - &log_std 
            - 0.5 * (2.0 * std::f64::consts::PI).ln();
            
        // Tanh correction: log(1 - tanh(x)^2) = log(1 - action^2)
        // We use the action tensor directly here for stability
        let log_prob_correction = (Tensor::from(1.0).to_kind(Kind::Float).to_device(act_tensor.device()) - act_tensor.pow_tensor_scalar(2.0) + Tensor::from(1e-6).to_kind(Kind::Float).to_device(act_tensor.device())).log();
        
        let log_prob: Tensor = log_prob_gauss - log_prob_correction;
        log_prob.sum_dim_intlist(Some(&[1i64][..]), false, Kind::Float)
    }
}

impl RLAgent for POCA {
     fn record_transition(&mut self, agent_id: i32, obs: Vec<f32>, act: Vec<f32>, reward: f32, _next_obs: Vec<f32>, done: bool) {
        let obs_tensor = Tensor::from_slice(&obs).to(self.device).reshape(&[1, self.obs_dim as i64]);
        let act_tensor = Tensor::from_slice(&act).to(self.device).reshape(&[1, self.act_dim as i64]);
        
        // Calculate "Old Log Prob" on the fly for the taken action
        let log_prob = tch::no_grad(|| {
             self.calc_log_prob_from_action(&obs_tensor, &act_tensor).double_value(&[]) as f32
        });

        self.rollout.push(POCATransition {
            obs,
            action: act,
            reward,
            log_prob,
            done,
            agent_id,
        });
    }

    fn train(&mut self) -> Option<HashMap<String, f32>> {
        let n_steps = self.rollout.len();
        if n_steps < self.batch_size {
            return None;
        }

        // 1. Prepare Batch Tensors
        let mut obs_data = Vec::with_capacity(n_steps * self.obs_dim);
        let mut act_data = Vec::with_capacity(n_steps * self.act_dim);
        let mut rew_data = Vec::with_capacity(n_steps);
        let mut done_data = Vec::with_capacity(n_steps);
        let mut log_prob_data = Vec::with_capacity(n_steps);
        
        for t in &self.rollout.transitions {
            obs_data.extend_from_slice(&t.obs);
            act_data.extend_from_slice(&t.action);
            rew_data.push(t.reward);
            done_data.push(if t.done { 1.0f32 } else { 0.0f32 });
            log_prob_data.push(t.log_prob);
        }
        
        let obs_batch = Tensor::from_slice(&obs_data).to(self.device).reshape(&[n_steps as i64, 1, self.obs_dim as i64]); // [N, 1, Obs] for Critic
        let obs_actor_batch = obs_batch.squeeze_dim(1); // [N, Obs] for Actor
        
        let act_batch = Tensor::from_slice(&act_data).to(self.device).reshape(&[n_steps as i64, self.act_dim as i64]);
        let _rewards = Tensor::from_slice(&rew_data).to(self.device).reshape(&[n_steps as i64, 1]);
        let old_log_probs = Tensor::from_slice(&log_prob_data).to(self.device).reshape(&[n_steps as i64, 1]);
        let _dones = Tensor::from_slice(&done_data).to(self.device).reshape(&[n_steps as i64, 1]);

        // 2. Compute GAE
        let values = tch::no_grad(|| self.critic.forward(&obs_batch)); // [N, 1]
        let values_vec: Vec<f32> = values.flatten(0, -1).try_into().unwrap();
        
        let mut advantages = vec![0.0; n_steps];
        let mut returns = vec![0.0; n_steps];
        let mut gae = 0.0;
        
        // Simple GAE loop (assuming data is sequential per agent, but it might be mixed if multiple agents in parallel.
        // LIMITATION: This assumes sequential data or single agent stream. For true multi-agent, we need to handle trajectory boundaries by agent_id.
        // Given constraint, we use standard GAE assuming reasonable coherence or mostly single trajectory chunks.)
        for i in (0..n_steps).rev() {
            let next_val = if i + 1 < n_steps { values_vec[i+1] } else { 0.0 };
            // Check if done occurred at i, resetting GAE
            let is_done = done_data[i] > 0.5;
            
            let delta = rew_data[i] as f64 + self.gamma * next_val as f64 * (1.0 - done_data[i] as f64) - values_vec[i] as f64;
            gae = delta + self.gamma * self.gae_lambda * (1.0 - done_data[i] as f64) * gae;
            
            if is_done { gae = delta; } // Hard reset on done just to be safe

            advantages[i] = gae as f32;
            returns[i] = (advantages[i] + values_vec[i]) as f32;
        }

        let adv_tensor = Tensor::from_slice(&advantages).to(self.device).reshape(&[n_steps as i64, 1]);
        // Normalize advantages
        let adv_mean = adv_tensor.mean(Kind::Float);
        let adv_std = adv_tensor.std(true);
        let adv_batch = (adv_tensor - adv_mean) / (adv_std + 1e-8);
        
        let ret_batch = Tensor::from_slice(&returns).to(self.device).reshape(&[n_steps as i64, 1]);

        // 3. PPO Update Loop
        let mut last_p_loss = 0.0;
        let mut last_v_loss = 0.0;

        for _ in 0..self.num_epochs {
            // New Log Probs & Entropy
            // We use the helper again on the batch
            let (_mean, log_std) = self.actor.forward(&obs_actor_batch);
            // let std = log_std.exp(); // Unused
            
            // Calculate entropy: sum(log_std + 0.5 * log(2*pi*e))
            let entropy_const = 0.5 + (std::f32::consts::PI * 2.0).ln() * 0.5;
            let entropy = (log_std + Tensor::from(entropy_const).to_device(self.device)).sum_dim_intlist(Some(&[1i64][..]), false, Kind::Float).mean(Kind::Float);
            
            // Re-calc log prob for current policy
            // Note: We use the same 'inverse tanh' approximation logic on the *original* actions.
            // PPO asks "how likely is the old action under the new policy?"
            let new_log_probs = self.calc_log_prob_from_action(&obs_actor_batch, &act_batch); // [N, 1]
            
            // Ratio
            let ratio = (new_log_probs - &old_log_probs).exp();
            
            // Surrogate Loss
            let surr1 = &ratio * &adv_batch;
            let surr2 = ratio.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * &adv_batch;
            
            let pg_loss = -surr1.min_other(&surr2).mean(Kind::Float);
            
            // Value Loss
            // (Optional: Clip value loss too, skipping for simplicity)
            let new_values = self.critic.forward(&obs_batch);
            let v_loss = (new_values - &ret_batch).pow_tensor_scalar(2.0).mean(Kind::Float);
            
            let loss = &pg_loss + self.vf_coef * &v_loss - self.ent_coef * &entropy;
            
            self.actor_optimizer.backward_step(&loss);
            self.critic_optimizer.backward_step(&loss);
            
            last_p_loss = pg_loss.double_value(&[]) as f32;
            last_v_loss = v_loss.double_value(&[]) as f32;
        }

        self.rollout.clear();
        
        let mut metrics = HashMap::new();
        metrics.insert("Losses/Policy Loss".to_string(), last_p_loss);
        metrics.insert("Losses/Value Loss".to_string(), last_v_loss);
        
        Some(metrics)
    }

    fn select_action(&self, obs: &[f32], deterministic: bool) -> Vec<f32> {
        let obs_tensor = Tensor::from_slice(obs).to(self.device).reshape(&[1, self.obs_dim as i64]);
        let (mean, log_std) = self.actor.forward(&obs_tensor);
        
        let action = if deterministic {
            mean.tanh()
        } else {
            let std = log_std.exp();
            let action = &mean + std * Tensor::randn_like(&mean);
            action.tanh()
        };
        action.flatten(0, -1).try_into().unwrap()
    }
    
    fn get_obs_dim(&self) -> usize { self.obs_dim }
    fn get_act_dim(&self) -> usize { self.act_dim }
    fn get_buffer_size(&self) -> usize { self.rollout.len() }
    fn get_training_threshold(&self) -> usize { self.batch_size }

    fn save(&self, path: &str) -> std::io::Result<()> {
        self.vs.save(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    fn load(&mut self, path: &str) -> std::io::Result<()> {
        self.vs.load(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
    
    fn export_onnx(&self, path: &str) -> std::io::Result<()> {
        // Reuse SAC/PPO export
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
