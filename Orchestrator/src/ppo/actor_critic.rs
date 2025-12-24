use tch::{nn, Tensor, Kind};
use std::f32::consts::PI;

#[derive(Debug)]
pub struct PPOSharedActorCritic {
    pub common_linear: nn::Linear,
    pub actor_linear: nn::Linear,
    pub critic_linear: nn::Linear,
    pub log_std: Tensor,
}

impl PPOSharedActorCritic {
    pub fn new(vs: &nn::Path, obs_dim: i64, act_dim: i64, hidden_dim: i64) -> Self {
        let common_linear = nn::linear(vs / "common", obs_dim, hidden_dim, Default::default());
        let actor_linear = nn::linear(vs / "actor", hidden_dim, act_dim, Default::default());
        let critic_linear = nn::linear(vs / "critic", hidden_dim, 1, Default::default());
        
        let log_std = vs.var("log_std", &[act_dim], nn::Init::Const(0.0));

        PPOSharedActorCritic {
            common_linear,
            actor_linear,
            critic_linear,
            log_std,
        }
    }

    pub fn forward(&self, obs: &Tensor) -> (Tensor, Tensor, Tensor) {
        let x = obs.apply(&self.common_linear).relu();
        let mean = x.apply(&self.actor_linear).tanh(); // Added Tanh
        let value = x.apply(&self.critic_linear);
        
        // Log std expansion
        let batch_size = mean.size()[0];
        let std = self.log_std.exp().unsqueeze(0).expand(&[batch_size, -1], true);
        
        (mean, std, value)
    }

    pub fn sample(&self, obs: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (mean, std, value) = self.forward(obs); // Forward now includes Tanh on mean
        
        // In PPO, we typically sample around the mean and THEN tanh, 
        // OR sample around the pre-tanh mean. 
        // To match ML-Agents exactly and ensure stability:
        let normal_dist = Tensor::randn_like(&mean);
        let action = &mean + &std * normal_dist;
        
        // Log Prob calculation (Simple Gaussian on the already-bounded mean)
        let var = std.pow_tensor_scalar(2.0);
        let const_term = 0.5 * (2.0 * PI).ln();
        let log_prob: Tensor = -0.5 * (action.shallow_clone() - &mean).pow_tensor_scalar(2.0) / &var
            - std.log()
            - Tensor::from(const_term).to_device(std.device());
            
        let log_prob_sum = log_prob.sum_dim_intlist(Some(&[1i64][..]), false, Kind::Float);

        (action, log_prob_sum, value)
    }
}
