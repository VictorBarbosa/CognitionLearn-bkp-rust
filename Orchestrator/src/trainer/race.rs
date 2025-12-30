use std::sync::{Mutex, Condvar};
use std::collections::HashMap;
use crate::agent::AgentType;
use crate::trainer::settings::TrainerSettings;

#[derive(Debug, Clone, PartialEq)]
pub enum RaceStatus {
    Running,
    WaitingForOthers,
    Eliminated,
    Winner,
}

#[derive(Debug)]
pub struct RaceParticipant {
    pub id: String,
    pub current_step: usize,
    pub current_reward: f32,
    pub status: RaceStatus,
    pub algorithm: AgentType,
}

#[derive(Debug, Clone)]
pub struct RaceRoundResult {
    pub milestone_idx: usize,
    pub start_step: usize,
    pub end_step: usize,
    pub eliminated_id: String,
    pub winner_id: String,
}

pub struct RaceController {
    pub total_steps: usize,
    pub participants: Mutex<HashMap<String, RaceParticipant>>,
    pub barrier: Condvar, 
    pub active_participants: Mutex<Vec<String>>, 
    pub checkpoints: Vec<f32>, 
    pub next_checkpoint_idx: Mutex<usize>,
    pub prev_checkpoint_step: Mutex<usize>, // Added
    pub best_model_path: Mutex<Option<String>>,
    pub best_model_settings: Mutex<Option<TrainerSettings>>,
    pub adoption_assignments: Mutex<HashMap<String, Vec<String>>>, // Winner -> [Adopted IDs]
}

impl RaceController {
    pub fn new(total_steps: usize, participant_ids: Vec<String>) -> Self {
        let mut participants = HashMap::new();
        for id in &participant_ids {
            participants.insert(id.clone(), RaceParticipant {
                id: id.clone(),
                current_step: 0,
                current_reward: -9999.0, 
                status: RaceStatus::Running,
                algorithm: AgentType::PPO, 
            });
        }

        // Dynamic Checkpoints: N agents -> N-1 eliminations.
        // If N=4: Checkpoints at 25% (eliminate 1), 50% (eliminate 1), 75% (eliminate 1 -> Winner).
        // Formula: k/N for k=1..N-1.
        let n = participant_ids.len();
        let mut checkpoints = Vec::new();
        if n > 1 {
            for i in 1..n {
                checkpoints.push(i as f32 / n as f32);
            }
        }
        
        println!("🏁 Race Configured. Participants: {}. Checkpoints: {:?}", n, checkpoints);

        Self {
            total_steps,
            participants: Mutex::new(participants),
            barrier: Condvar::new(),
            active_participants: Mutex::new(participant_ids),
            checkpoints,
            next_checkpoint_idx: Mutex::new(0),
            prev_checkpoint_step: Mutex::new(0),
            best_model_path: Mutex::new(None),
            best_model_settings: Mutex::new(None),
            adoption_assignments: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_adopted_channels(&self, winner_id: &str) -> Option<Vec<String>> {
        let mut assignments = self.adoption_assignments.lock().unwrap();
        assignments.remove(winner_id)
    }

    pub fn report_progress(
        &self, 
        id: &str, 
        step: usize, 
        reward: f32, 
        algo: AgentType, 
        model_path: &str, 
        settings: &TrainerSettings
    ) -> (bool, Option<(String, TrainerSettings)>, Option<RaceRoundResult>) {
        let mut parts = self.participants.lock().unwrap();
        let mut next_ckpt_idx = self.next_checkpoint_idx.lock().unwrap();
        
        if let Some(p) = parts.get_mut(id) {
            p.current_step = step;
            p.current_reward = reward; 
            p.algorithm = algo;
        }
        
        // Track Best Global Reward to capture the winner's state even before elimination logic triggers
        let current_best_reward = parts.values()
            .filter(|p| p.status != RaceStatus::Eliminated)
            .map(|p| p.current_reward)
            .fold(-f32::INFINITY, f32::max);

        if reward >= current_best_reward {
            let mut best_path = self.best_model_path.lock().unwrap();
            let mut best_settings = self.best_model_settings.lock().unwrap();
            *best_path = Some(model_path.to_string());
            *best_settings = Some(settings.clone());
        }

        if *next_ckpt_idx >= self.checkpoints.len() {
            return (true, None, None); 
        }

        let threshold_step = (self.total_steps as f32 * self.checkpoints[*next_ckpt_idx]) as usize;

        if step >= threshold_step {
            if let Some(p) = parts.get_mut(id) {
                p.status = RaceStatus::WaitingForOthers;
            }

            let active = self.active_participants.lock().unwrap();
            let all_arrived = active.iter().all(|aid| {
                if let Some(p) = parts.get(aid) {
                    p.current_step >= threshold_step
                } else {
                    false
                }
            });

            if all_arrived {
                println!("🏁 Race Checkpoint {:.0}% reached!", self.checkpoints[*next_ckpt_idx] * 100.0);
                
                let mut ranked: Vec<(String, f32)> = active.iter()
                    .filter_map(|aid| parts.get(aid).map(|p| (aid.clone(), p.current_reward)))
                    .collect();
                
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                let mut round_result = None;

                if ranked.len() > 1 {
                    let (loser_id, loser_reward) = ranked.last().unwrap().clone();
                    let (winner_id, winner_reward) = ranked.first().unwrap().clone();
                    
                    println!("❌ Eliminating {} (Reward: {:.2}). Leader is {} (Reward: {:.2})", loser_id, loser_reward, winner_id, winner_reward);
                    
                    // Create Result
                    let mut prev_step_guard = self.prev_checkpoint_step.lock().unwrap();
                    round_result = Some(RaceRoundResult {
                        milestone_idx: *next_ckpt_idx,
                        start_step: *prev_step_guard,
                        end_step: threshold_step,
                        eliminated_id: loser_id.clone(),
                        winner_id: winner_id.clone(),
                    });
                    *prev_step_guard = threshold_step;

                    if let Some(p) = parts.get_mut(&loser_id) {
                        p.status = RaceStatus::Eliminated;
                    }
                    
                    let active_count_after = ranked.len() - 1;
                    
                    if active_count_after == 1 {
                        println!("🏆 Finalist determined: {}! Consolidating environments for Victory Lap!", winner_id);
                        
                        // Set winner status
                        if let Some(p) = parts.get_mut(&winner_id) {
                            p.status = RaceStatus::Winner;
                        }
                        
                        // Consolidation Phase: Winner adopts everyone else
                        let all_ids: Vec<String> = parts.keys().cloned().collect();
                        let adopted: Vec<String> = all_ids.into_iter().filter(|x| x != &winner_id).collect();
                        
                        let mut assignments = self.adoption_assignments.lock().unwrap();
                        assignments.insert(winner_id.clone(), adopted);

                        *next_ckpt_idx += 1;
                        self.barrier.notify_all();
                        
                        return (true, None, round_result);
                    } else {
                        // Eliminate loser from active set
                        let mut active_guard = self.active_participants.lock().unwrap();
                        active_guard.retain(|x| x != &loser_id);
                    }
                }

                *next_ckpt_idx += 1;
                self.barrier.notify_all();
                
                // If round_result was created (elimination happened), pass it.
                // We must return here to avoid falling into wait() logic below.
                return (true, None, round_result); 
            } else {
                // Wait and re-acquire lock
                parts = self.barrier.wait(parts).unwrap();
            }
        }
        
        // At this point, we hold the lock 'parts'.
        if let Some(p) = parts.get(id) {
            match p.status {
                RaceStatus::Eliminated => {
                    println!("💤 Agent {} sleeping (Eliminated)...", id);
                    
                    loop {
                        parts = self.barrier.wait(parts).unwrap();
                        
                        // Check if status changed to Running (Respawn - for previous phases)
                        if let Some(me) = parts.get(id) {
                            if me.status == RaceStatus::Running {
                                println!("⚡ Agent {} Respawned!", id);
                                let winner_path = self.best_model_path.lock().unwrap().clone();
                                let winner_settings = self.best_model_settings.lock().unwrap().clone();
                                
                                if let (Some(path), Some(set)) = (winner_path, winner_settings) {
                                    return (true, Some((path, set)), None);
                                }
                                return (true, None, None);
                            }
                        }
                        
                        // Check if race entered Final Phase (Consolidation) - Losers should die
                        let idx = self.next_checkpoint_idx.lock().unwrap();
                        if *idx >= self.checkpoints.len() {
                             println!("💀 Agent {} terminating (Final Consolidation).", id);
                             return (false, None, None); // Die
                        }
                    }
                },
                RaceStatus::Winner => (true, None, None),
                RaceStatus::Running | RaceStatus::WaitingForOthers => (true, None, None),
            }
        } else {
            (false, None, None)
        }
    }
}