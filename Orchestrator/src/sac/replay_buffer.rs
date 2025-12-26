use tch::{Tensor, Device, Kind};

pub struct Batch {
    pub obs: Tensor,
    pub actions: Tensor,
    pub rewards: Tensor,
    pub next_obs: Tensor,
    pub done: Tensor,
}

pub struct ReplayBuffer {
    capacity: usize,
    device: Device,
    obs_dim: usize,
    act_dim: usize,
    
    // Contiguous memory on GPU
    obs_buf: Tensor,
    action_buf: Tensor,
    reward_buf: Tensor,
    next_obs_buf: Tensor,
    done_buf: Tensor,
    
    ptr: usize,
    size: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, obs_dim: usize, act_dim: usize, device: Device) -> Self {
        // Pre-allocate tensors on the target device (GPU/MPS/CPU)
        let obs_buf = Tensor::zeros(&[capacity as i64, obs_dim as i64], (Kind::Float, device));
        let action_buf = Tensor::zeros(&[capacity as i64, act_dim as i64], (Kind::Float, device));
        let reward_buf = Tensor::zeros(&[capacity as i64, 1], (Kind::Float, device));
        let next_obs_buf = Tensor::zeros(&[capacity as i64, obs_dim as i64], (Kind::Float, device));
        let done_buf = Tensor::zeros(&[capacity as i64, 1], (Kind::Float, device));

        Self {
            capacity,
            device,
            obs_dim,
            act_dim,
            obs_buf,
            action_buf,
            reward_buf,
            next_obs_buf,
            done_buf,
            ptr: 0,
            size: 0,
        }
    }

    pub fn push(
        &mut self, 
        obs: &[f32], 
        actions: &[f32], 
        reward: f32, 
        next_obs: &[f32], 
        done: bool
    ) {
        let idx = self.ptr as i64;
        
        // Non-blocking copies to device
        tch::no_grad(|| {
            self.obs_buf.narrow(0, idx, 1).copy_(&Tensor::from_slice(obs).to_device(self.device).reshape(&[1, self.obs_dim as i64]));
            self.action_buf.narrow(0, idx, 1).copy_(&Tensor::from_slice(actions).to_device(self.device).reshape(&[1, self.act_dim as i64]));
            self.reward_buf.narrow(0, idx, 1).fill_(reward as f64);
            self.next_obs_buf.narrow(0, idx, 1).copy_(&Tensor::from_slice(next_obs).to_device(self.device).reshape(&[1, self.obs_dim as i64]));
            self.done_buf.narrow(0, idx, 1).fill_(if done { 0.0 } else { 1.0 }); // SAC uses (1-done) in formula
        });

        self.ptr = (self.ptr + 1) % self.capacity;
        self.size = std::cmp::min(self.size + 1, self.capacity);
    }

    pub fn sample(&self, batch_size: usize) -> Batch {
        // Random indices generated on GPU
        let indices = Tensor::randint(self.size as i64, &[batch_size as i64], (Kind::Int64, self.device));

        Batch {
            obs: self.obs_buf.index_select(0, &indices),
            actions: self.action_buf.index_select(0, &indices),
            rewards: self.reward_buf.index_select(0, &indices),
            next_obs: self.next_obs_buf.index_select(0, &indices),
            done: self.done_buf.index_select(0, &indices),
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }
}