use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor, Kind};

pub struct ICM {
    pub feature_encoder: nn::Sequential,
    pub forward_model: nn::Sequential,
    pub inverse_model: nn::Sequential,
    pub optimizer: nn::Optimizer,
    pub strength: f64,
    pub beta: f64, // Weight of forward loss vs inverse loss (usually 0.2)
    pub device: Device,
}

impl ICM {
    pub fn new(
        vs_full: &nn::VarStore,
        p: &nn::Path,
        obs_dim: i64,
        act_dim: i64,
        hidden_dim: i64,
        strength: f64,
        learning_rate: f64,
        device: Device,
    ) -> Self {
        let feature_dim = hidden_dim; // Encode obs to this dim

        // 1. Feature Encoder: s -> phi(s)
        let feature_encoder = nn::seq()
            .add(nn::linear(p / "encoder_l1", obs_dim, hidden_dim, Default::default()))
            .add(nn::func(|xs| xs.relu()))
            .add(nn::linear(p / "encoder_l2", hidden_dim, feature_dim, Default::default())); // No activation at output

        // 2. Forward Model: (phi(s), a) -> phi(s') (Predicts next state features)
        let forward_model = nn::seq()
            .add(nn::linear(p / "fwd_l1", feature_dim + act_dim, hidden_dim, Default::default()))
            .add(nn::func(|xs| xs.relu()))
            .add(nn::linear(p / "fwd_l2", hidden_dim, feature_dim, Default::default()));

        // 3. Inverse Model: (phi(s), phi(s')) -> a (Predicts action taken)
        let inverse_model = nn::seq()
            .add(nn::linear(p / "inv_l1", feature_dim * 2, hidden_dim, Default::default()))
            .add(nn::func(|xs| xs.relu()))
            .add(nn::linear(p / "inv_l2", hidden_dim, act_dim, Default::default()));

        let optimizer = nn::Adam::default().build(vs_full, learning_rate).expect("Failed to build ICM optimizer");

        Self {
            feature_encoder,
            forward_model,
            inverse_model,
            optimizer,
            strength,
            beta: 0.2, // Standard value
            device,
        }
    }

    /// Returns intrinsic rewards for a batch of transitions
    /// obs: [B, ObsDim], next_obs: [B, ObsDim], actions: [B, ActDim]
    pub fn compute_intrinsic_reward(&self, obs: &Tensor, next_obs: &Tensor, actions: &Tensor) -> Tensor {
        // Encode states
        let phi_s = self.feature_encoder.forward(obs);
        let phi_next_s = self.feature_encoder.forward(next_obs);

        // Forward Model Prediction: Predict phi(s') from phi(s) and a
        let fwd_input = Tensor::cat(&[&phi_s, actions], 1);
        let pred_phi_next_s = self.forward_model.forward(&fwd_input);

        // Intrinsic Reward is the MSE Error of the Forward Model (Curiosity)
        // We calculate (pred - target)^2 sum over features, scaled by 0.5
        let forward_loss = (pred_phi_next_s - phi_next_s).pow_tensor_scalar(2.0).sum_dim_intlist(Some(&[1i64][..]), false, Kind::Float) * 0.5;
        
        forward_loss * self.strength
    }

    /// Updates the ICM networks
    pub fn update(&mut self, obs: &Tensor, next_obs: &Tensor, actions: &Tensor) -> f32 {
        let phi_s = self.feature_encoder.forward(obs);
        let phi_next_s = self.feature_encoder.forward(next_obs);

        // 1. Forward Loss (Prediction Error)
        let fwd_input = Tensor::cat(&[&phi_s, actions], 1);
        let pred_phi_next_s = self.forward_model.forward(&fwd_input);
        let forward_loss = (pred_phi_next_s - &phi_next_s).pow_tensor_scalar(2.0).mean(Kind::Float);

        // 2. Inverse Loss (Action Prediction Error)
        let inv_input = Tensor::cat(&[&phi_s, &phi_next_s], 1);
        let pred_actions = self.inverse_model.forward(&inv_input);
        // Assuming continuous actions, we use MSE. For discrete, CrossEntropy.
        let inverse_loss = (pred_actions - actions).pow_tensor_scalar(2.0).mean(Kind::Float);

        // Total Loss = (1 - beta) * InverseLoss + beta * ForwardLoss
        let total_loss = inverse_loss * (1.0 - self.beta) + forward_loss * self.beta;

        self.optimizer.backward_step(&total_loss);

        total_loss.double_value(&[]) as f32
    }
}
