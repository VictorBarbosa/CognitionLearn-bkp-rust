use tch::{nn, nn::Module, Tensor};

#[derive(Debug)]
pub struct Critic {
    pub net: nn::Sequential,
}

impl Critic {
    pub fn new(vs: &nn::Path, obs_dim: i64, act_dim: i64, hidden_dim: i64) -> Self {
        let linear1 = nn::linear(vs / "linear1", obs_dim + act_dim, hidden_dim, Default::default());
        let linear2 = nn::linear(vs / "linear2", hidden_dim, hidden_dim, Default::default());
        let output = nn::linear(vs / "output", hidden_dim, 1, Default::default());

        let net = nn::seq()
            .add(linear1)
            .add_fn(|xs| xs.relu())
            .add(linear2)
            .add_fn(|xs| xs.relu())
            .add(output);
            
        Critic { net }
    }

    pub fn forward(&self, obs: &Tensor, act: &Tensor) -> Tensor {
        let x = Tensor::cat(&[obs, act], 1);
        self.net.forward(&x)
    }
}
