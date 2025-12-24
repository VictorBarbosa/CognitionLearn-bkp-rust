use tch::{nn, Tensor};

pub struct Critic {
    pub l1: nn::Linear,
    pub l2: nn::Linear,
    pub l3: nn::Linear,
}

impl Critic {
    pub fn new(p: &nn::Path, obs_dim: i64, act_dim: i64, hidden_dim: i64) -> Self {
        let l1 = nn::linear(p / "l1", obs_dim + act_dim, hidden_dim, Default::default());
        let l2 = nn::linear(p / "l2", hidden_dim, hidden_dim, Default::default());
        let l3 = nn::linear(p / "l3", hidden_dim, 1, Default::default());
        
        Self { l1, l2, l3 }
    }

    pub fn forward(&self, obs: &Tensor, act: &Tensor) -> Tensor {
        let x = Tensor::cat(&[obs, act], 1);
        let x = x.apply(&self.l1).relu();
        let x = x.apply(&self.l2).relu();
        x.apply(&self.l3)
    }
}
