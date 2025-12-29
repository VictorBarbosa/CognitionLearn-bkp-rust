pub mod actor_critic;
pub mod rollout_buffer;
pub mod icm;

use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use crate::ppo::actor_critic::PPOSharedActorCritic;
use crate::ppo::rollout_buffer::{Rollout, PPOTransition};
use crate::ppo::icm::ICM;
use crate::agent::RLAgent;
use std::collections::HashMap;

pub struct PPO {
    pub model: PPOSharedActorCritic,
    pub vs: nn::VarStore,
    pub optimizer: nn::Optimizer,
    pub rollout: Rollout,
    pub device: Device,
    pub gamma: f64,
    pub lambda: f64,
    pub eps_clip: f64,
    pub entropy_coef: f64,
    pub num_epochs: usize,
    pub horizon: usize,
    pub minibatch_size: usize,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub train_count: usize,
    pub sensor_sizes: Option<Vec<i64>>,
    pub max_grad_norm: f64,
    
    // Adaptive Entropy (PPO_ET)
    pub use_adaptive_entropy: bool,
    pub log_alpha: Option<Tensor>,
    pub alpha_optimizer: Option<nn::Optimizer>,
    pub target_entropy: f64,

    // Curiosity (PPO_CE)
    pub icm: Option<ICM>,
}

impl PPO {
    pub fn new(
        obs_dim: usize, 
        act_dim: usize, 
        hidden_dim: usize, 
        learning_rate: f64,
        gamma: f64,
        gae_lambda: f64,
        clip_ratio: f64,
        entropy_coef: f64,
        num_epochs: usize,
        horizon: usize,
        minibatch_size: usize,
        max_grad_norm: f64,
        device: Device,
        sensor_sizes: Option<Vec<i64>>,
        // New flags
        use_adaptive_entropy: bool,
        use_curiosity: bool,
        curiosity_strength: f64,
        curiosity_learning_rate: f64,
        _memory_size: Option<usize>, // Placeholder
        _sequence_length: Option<usize>, // Placeholder
    ) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root();
        
        let model = PPOSharedActorCritic::new(&root, obs_dim as i64, act_dim as i64, hidden_dim as i64);
        let optimizer = nn::Adam::default().build(&vs, learning_rate).expect("Failed to build optimizer");

        // Adaptive Entropy Init
        let (log_alpha, alpha_optimizer) = if use_adaptive_entropy {
            let la = root.var("log_alpha", &[1], nn::Init::Const(entropy_coef.ln())); // Init with config value
            let opt = nn::Adam::default().build(&vs, learning_rate).expect("Alpha opt");
            (Some(la), Some(opt))
        } else {
            (None, None)
        };
        let target_entropy = -(act_dim as f64); // Heuristic target

        // Curiosity Init
        let icm = if use_curiosity {
            Some(ICM::new(
                &vs, // Pass VarStore
                &(&root / "icm"),
                obs_dim as i64,
                act_dim as i64,
                hidden_dim as i64,
                curiosity_strength,
                curiosity_learning_rate,
                device
            ))
        } else {
            None
        };

        Self {
            model,
            vs,
            optimizer,
            rollout: Rollout::new(),
            device,
            gamma,
            lambda: gae_lambda,
            eps_clip: clip_ratio,
            entropy_coef,
            num_epochs,
            horizon,
            minibatch_size,
            obs_dim,
            act_dim,
            train_count: 0,
            sensor_sizes,
            max_grad_norm,
            use_adaptive_entropy,
            log_alpha,
            alpha_optimizer,
            target_entropy,
            icm,
        }
    }
}

impl RLAgent for PPO {
    fn record_transition(&mut self, agent_id: i32, obs: Vec<f32>, act: Vec<f32>, reward: f32, next_obs: Vec<f32>, done: bool) {
        let obs_tensor = Tensor::from_slice(&obs).to(self.device).reshape(&[1, self.obs_dim as i64]);
        
        // OPTIMIZATION: Single forward pass to get both Value and Action Distribution parameters
        let (mean, std, value) = tch::no_grad(|| self.model.forward(&obs_tensor));
        
        let act_tensor = Tensor::from_slice(&act).to(self.device).reshape(&[1, self.act_dim as i64]);
        
        let var = std.pow_tensor_scalar(2.0);
        let log_prob: Tensor = -0.5 * (act_tensor - &mean).pow_tensor_scalar(2.0) / &var
            - std.log()
            - 0.5 * (2.0 * std::f64::consts::PI).ln();
        let log_prob_val = log_prob.sum_dim_intlist(Some(&[1i64][..]), false, Kind::Float).double_value(&[]) as f32;
        let value_val = value.double_value(&[]) as f32;

        self.rollout.add(PPOTransition {
            agent_id, // Pass correct ID
            observation: obs,
            action: act,
            reward,
            next_observation: next_obs,
            log_prob: log_prob_val,
            done,
            value: value_val,
        });
    }

    fn train(&mut self) -> Option<HashMap<String, f32>> {
        if self.rollout.len() < self.horizon {
            return None;
        }

        // --- 1. Bootstrapping Phase (Pre-GAE) ---
        let mut bootstrap_map: HashMap<i32, f32> = HashMap::new();
        {
            let mut obs_to_bootstrap = Vec::new();
            let mut ids_to_bootstrap = Vec::new();

            for (agent_id, transitions) in &self.rollout.agent_transitions {
                if let Some(last_t) = transitions.last() {
                    if !last_t.done {
                         obs_to_bootstrap.extend_from_slice(&last_t.next_observation);
                         ids_to_bootstrap.push(*agent_id);
                    }
                }
            }

            if !ids_to_bootstrap.is_empty() {
                let n_boot = ids_to_bootstrap.len() as i64;
                let obs_tensor = Tensor::from_slice(&obs_to_bootstrap).to(self.device).reshape(&[n_boot, self.obs_dim as i64]);
                let (_, _, values) = tch::no_grad(|| self.model.forward(&obs_tensor));
                let values_vec: Vec<f32> = values.flatten(0, -1).try_into().unwrap();

                for (i, &agent_id) in ids_to_bootstrap.iter().enumerate() {
                    bootstrap_map.insert(agent_id, values_vec[i]);
                }
            }
        }

        // --- 2. GAE Calculation ---
        let mut all_obs = Vec::new();
        let mut all_acts = Vec::new();
        let mut all_log_probs = Vec::new();
        let mut all_advantages = Vec::new();
        let mut all_returns = Vec::new();
        let mut all_values = Vec::new(); 
        let mut all_next_obs = Vec::new(); 

        for (agent_id, transitions) in &self.rollout.agent_transitions {
            let n = transitions.len();
            let mut advantages = vec![0.0; n];
            let mut returns = vec![0.0; n];
            
            let mut gae = 0.0;
            for i in (0..n).rev() {
                let t = &transitions[i];
                
                // Get Next Value (Bootstrap or stored)
                let next_val = if i + 1 < n {
                    transitions[i+1].value
                } else {
                    *bootstrap_map.get(agent_id).unwrap_or(&0.0)
                };

                let delta = t.reward as f64 + self.gamma * next_val as f64 * (if t.done { 0.0 } else { 1.0 }) - t.value as f64;
                gae = delta + self.gamma * self.lambda * (if t.done { 0.0 } else { 1.0 }) * gae;
                
                advantages[i] = gae as f32;
                returns[i] = (advantages[i] + t.value) as f32;
            }

            for (i, t) in transitions.iter().enumerate() {
                all_obs.extend_from_slice(&t.observation);
                all_acts.extend_from_slice(&t.action);
                all_log_probs.push(t.log_prob);
                all_advantages.push(advantages[i]);
                all_returns.push(returns[i]);
                all_values.push(t.value);
                all_next_obs.extend_from_slice(&t.next_observation);
            }
        }

        let n_total = all_obs.len() / self.obs_dim;
        let obs_full = Tensor::from_slice(&all_obs).to(self.device).reshape(&[n_total as i64, self.obs_dim as i64]);
        let act_full = Tensor::from_slice(&all_acts).to(self.device).reshape(&[n_total as i64, self.act_dim as i64]);
        let log_probs_full = Tensor::from_slice(&all_log_probs).to(self.device).reshape(&[n_total as i64, 1]);
        let adv_full = Tensor::from_slice(&all_advantages).to(self.device).reshape(&[n_total as i64, 1]);
        let ret_full = Tensor::from_slice(&all_returns).to(self.device).reshape(&[n_total as i64, 1]);
        let old_values_full = Tensor::from_slice(&all_values).to(self.device).reshape(&[n_total as i64, 1]);
        let next_obs_full = Tensor::from_slice(&all_next_obs).to(self.device).reshape(&[n_total as i64, self.obs_dim as i64]);

        let adv_mean = adv_full.mean(Kind::Float);
        let adv_std = adv_full.std(true);
        let adv_normalized = (adv_full - adv_mean) / (adv_std + 1e-8);
        
        let mut last_p_loss = 0.0;
        let mut last_v_loss = 0.0;
        let mut last_entropy = 0.0;
        let mut last_alpha = self.entropy_coef as f32;
        let mut last_icm_loss = 0.0;
        
        let num_samples = n_total as i64;
        let batch_size = self.minibatch_size;

        for _ in 0..self.num_epochs {
            let indices = Tensor::randperm(num_samples, (Kind::Int64, self.device));
            
            for start in (0..num_samples).step_by(batch_size) {
                let end = (start + batch_size as i64).min(num_samples);
                let idx = indices.narrow(0, start, end - start);
                
                let obs_mini = obs_full.index_select(0, &idx);
                let act_mini = act_full.index_select(0, &idx);
                let ret_mini = ret_full.index_select(0, &idx);
                let adv_mini = adv_normalized.index_select(0, &idx);
                let old_log_probs_mini = log_probs_full.index_select(0, &idx);
                let old_values_mini = old_values_full.index_select(0, &idx);
                let next_obs_mini = next_obs_full.index_select(0, &idx);

                let (mean, std, values) = self.model.forward(&obs_mini);
                
                let var = std.pow_tensor_scalar(2.0);
                let log_prob: Tensor = -0.5 * (&act_mini - &mean).pow_tensor_scalar(2.0) / &var
                    - std.log()
                    - Tensor::from(0.5 * (2.0 * std::f32::consts::PI).ln()).to_device(self.device);
                let log_prob = log_prob.sum_dim_intlist(Some(&[1i64][..]), false, Kind::Float).reshape(&[idx.size()[0], 1]);

                let entropy_const = 0.5 + (std::f32::consts::PI * 2.0).ln() * 0.5;
                let entropy = (std.log() + Tensor::from(entropy_const).to_device(self.device)).sum_dim_intlist(Some(&[1i64][..]), false, Kind::Float).mean(Kind::Float);
                
                let ratio = (log_prob - &old_log_probs_mini).exp();
                let surr1 = &ratio * &adv_mini;
                let surr2 = ratio.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * &adv_mini;
                
                let policy_loss = -surr1.min_other(&surr2).mean(Kind::Float);
                
                // --- Value Clipping ---
                let v_clipped = &old_values_mini + (&values - &old_values_mini).clamp(-self.eps_clip, self.eps_clip);
                let v_loss1 = (&values - &ret_mini).pow_tensor_scalar(2.0);
                let v_loss2 = (v_clipped - &ret_mini).pow_tensor_scalar(2.0);
                let value_loss = v_loss1.max_other(&v_loss2).mean(Kind::Float);
                
                let current_alpha = if self.use_adaptive_entropy {
                    if let (Some(log_alpha), Some(opt)) = (&self.log_alpha, &mut self.alpha_optimizer) {
                        let alpha_loss = -(log_alpha * (entropy.detach() + self.target_entropy)).mean(Kind::Float);
                        opt.backward_step(&alpha_loss);
                        log_alpha.exp().double_value(&[])
                    } else { self.entropy_coef }
                } else { self.entropy_coef };
                
                let loss = &policy_loss + &value_loss * 0.5 - &entropy * current_alpha;
                
                // --- Gradient Clipping ---
                self.optimizer.backward_step_clip(&loss, self.max_grad_norm);
                
                if let Some(icm) = &mut self.icm {
                    last_icm_loss = icm.update(&obs_mini, &next_obs_mini, &act_mini);
                }
                
                last_p_loss = policy_loss.double_value(&[]) as f32;
                last_v_loss = value_loss.double_value(&[]) as f32;
                last_entropy = entropy.double_value(&[]) as f32;
                last_alpha = current_alpha as f32;
            }
        }

        self.rollout.clear();
        self.train_count += 1;

        let mut metrics = HashMap::new();
        metrics.insert("Losses/Policy Loss".to_string(), last_p_loss);
        metrics.insert("Losses/Value Loss".to_string(), last_v_loss);
        metrics.insert("Policy/Entropy".to_string(), last_entropy);
        if self.use_adaptive_entropy { metrics.insert("Policy/Alpha".to_string(), last_alpha); }
        if self.icm.is_some() { metrics.insert("Curiosity/ICM Loss".to_string(), last_icm_loss); }
        Some(metrics)
    }

    fn select_action(&self, obs: &[f32], deterministic: bool) -> Vec<f32> {
        let obs_tensor = Tensor::from_slice(obs).to(self.device).reshape(&[1, self.obs_dim as i64]);
        
        let action = if deterministic {
            let (mean, _, _) = tch::no_grad(|| self.model.forward(&obs_tensor));
            mean // forward already applies Tanh
        } else {
            let (action, _, _) = tch::no_grad(|| self.model.sample(&obs_tensor));
            action // sample already applies Tanh
        };
        
        let action_flat = action.flatten(0, -1);
        action_flat.try_into().unwrap()
    }

    fn get_obs_dim(&self) -> usize { self.obs_dim }
    fn get_act_dim(&self) -> usize { self.act_dim }
    fn get_buffer_size(&self) -> usize { self.rollout.len() }
    fn get_training_threshold(&self) -> usize { self.horizon }

    fn save(&self, path: &str) -> std::io::Result<()> {
        self.vs.save(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    fn load(&mut self, path: &str) -> std::io::Result<()> {
        self.vs.load(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    fn export_onnx(&self, path: &str) -> std::io::Result<()> {
        // Access weights
        let common_weight = &self.model.common_linear.ws;
        let common_bias = self.model.common_linear.bs.as_ref().expect("Common bias missing");
        let actor_weight = &self.model.actor_linear.ws;
        let actor_bias = self.model.actor_linear.bs.as_ref().expect("Actor bias missing");

        crate::onnx_utils::export_ppo_onnx(
            common_weight,
            common_bias,
            actor_weight,
            actor_bias,
            self.obs_dim as i64,
            self.act_dim as i64,
            common_weight.size()[0], 
            path,
            &self.sensor_sizes
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}
