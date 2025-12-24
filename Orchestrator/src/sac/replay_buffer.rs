
use rand::prelude::IndexedRandom;
use std::collections::VecDeque;
use tch::{Tensor, Device};

pub struct Transition {
    pub obs: Vec<f32>,
    pub actions: Vec<f32>,
    pub reward: f32,
    pub next_obs: Vec<f32>,
    pub done: bool,
}

pub struct Batch {
    pub obs: Tensor,
    pub actions: Tensor,
    pub rewards: Tensor,
    pub next_obs: Tensor,
    pub done: Tensor,
}

pub struct ReplayBuffer {
    capacity: usize,
    pub buffer: VecDeque<Transition>,
    obs_dim: usize,
    act_dim: usize,
    device: Device,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, obs_dim: usize, act_dim: usize, device: Device) -> Self {
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
            obs_dim,
            act_dim,
            device,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(transition);
    }

    pub fn sample(&self, batch_size: usize) -> Batch {
        let mut rng = rand::rng();
        let indices: Vec<usize> = (0..self.buffer.len()).collect();
        let sampled_indices = indices.choose_multiple(&mut rng, batch_size);
        
        let mut obs_vec = Vec::with_capacity(batch_size * self.obs_dim);
        let mut acts_vec = Vec::with_capacity(batch_size * self.act_dim);
        let mut rews_vec = Vec::with_capacity(batch_size);
        let mut next_obs_vec = Vec::with_capacity(batch_size * self.obs_dim);
        let mut done_vec = Vec::with_capacity(batch_size);

        for &i in sampled_indices {
            let t = &self.buffer[i];
            obs_vec.extend_from_slice(&t.obs);
            acts_vec.extend_from_slice(&t.actions);
            rews_vec.push(t.reward);
            next_obs_vec.extend_from_slice(&t.next_obs);
            done_vec.push(if t.done { 0.0f32 } else { 1.0f32 });
        }

        let obs = Tensor::from_slice(&obs_vec).to(self.device).reshape(&[batch_size as i64, self.obs_dim as i64]);
        let actions = Tensor::from_slice(&acts_vec).to(self.device).reshape(&[batch_size as i64, self.act_dim as i64]);
        let rewards = Tensor::from_slice(&rews_vec).to(self.device).reshape(&[batch_size as i64, 1]);
        let next_obs = Tensor::from_slice(&next_obs_vec).to(self.device).reshape(&[batch_size as i64, self.obs_dim as i64]);
        let done = Tensor::from_slice(&done_vec).to(self.device).reshape(&[batch_size as i64, 1]);

        Batch {
            obs,
            actions,
            rewards,
            next_obs,
            done,
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}
