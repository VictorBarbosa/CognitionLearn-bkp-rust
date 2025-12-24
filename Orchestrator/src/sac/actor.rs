use tch::{nn, nn::Module, Tensor, Kind};
use std::f64::consts::PI;

#[derive(Debug)]
pub struct Actor {
    pub l1: nn::Linear,
    pub l2: nn::Linear,
    pub mean: nn::Linear,
    pub log_std: nn::Linear,
}

impl Actor {
    pub fn new(vs: &nn::Path, input_dim: i64, hidden_dim: i64, output_dim: i64) -> Self {
        let l1 = nn::linear(vs / "linear1", input_dim, hidden_dim, Default::default());
        let l2 = nn::linear(vs / "linear2", hidden_dim, hidden_dim, Default::default());
        let mean = nn::linear(vs / "mean", hidden_dim, output_dim, Default::default());
        let log_std = nn::linear(vs / "log_std", hidden_dim, output_dim, Default::default());

        Actor { l1, l2, mean, log_std }
    }

    pub fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let x = xs.apply(&self.l1).relu();
        let features = x.apply(&self.l2).relu();
        
        let mean = self.mean.forward(&features);
        let log_std = self.log_std.forward(&features);
        
        // Clamp log_std for numerical stability [-20, 2]
        let log_std = log_std.clamp(-20.0, 2.0);
        
        (mean, log_std)
    }

    pub fn sample(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let (mean, log_std) = self.forward(xs);
        let std = log_std.exp();
        
        let epsilon = Tensor::randn_like(&mean);
        let x_t = &mean + &std * &epsilon;
        let action = x_t.tanh();
        
        let log_prob = self.calc_log_prob(&mean, &log_std, &x_t);
        
        (action, log_prob)
    }

    fn calc_log_prob(&self, mean: &Tensor, log_std: &Tensor, x_t: &Tensor) -> Tensor {
        let std = log_std.exp();
        let var = std.pow_tensor_scalar(2.0);
        
        // Gaussian log prob: -0.5 * ((x - mean) / std)^2 - log(std) - 0.5 * log(2 * pi)
        let log_prob_gauss = -0.5 * (x_t - mean).pow_tensor_scalar(2.0) / &var
            - log_std 
            - 0.5 * (2.0 * PI).ln();
        
        // Tanh squashing correction: log(1 - tanh(x)^2)
        // x_t is pre-tanh, action is tanh(x_t)
        let action = x_t.tanh();
        
        // Using formula: log(1 - a^2 + epsilon)
        let log_prob_correction = (Tensor::from(1.0).to_kind(Kind::Float).to_device(action.device()) - action.pow_tensor_scalar(2.0) + Tensor::from(1e-6).to_kind(Kind::Float).to_device(action.device())).log();
        
        let diff: Tensor = log_prob_gauss - log_prob_correction;
        diff.sum_dim_intlist(Some(&[1i64][..]), false, Kind::Float)
    }
}
