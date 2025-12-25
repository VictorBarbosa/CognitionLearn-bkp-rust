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

pub struct RaceController {
    pub total_steps: usize,
    pub participants: Mutex<HashMap<String, RaceParticipant>>,
    pub barrier: Condvar, 
    pub active_participants: Mutex<Vec<String>>, 
    pub checkpoints: Vec<f32>, 
    pub next_checkpoint_idx: Mutex<usize>,
    pub best_model_path: Mutex<Option<String>>,
    pub best_model_settings: Mutex<Option<TrainerSettings>>,
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
            best_model_path: Mutex::new(None),
            best_model_settings: Mutex::new(None),
        }
    }

    pub fn report_progress(
        &self, 
        id: &str, 
        step: usize, 
        reward: f32, 
        algo: AgentType, 
        model_path: &str,
        settings: &TrainerSettings
    ) -> (bool, Option<(String, TrainerSettings)>) {
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
            return (true, None); 
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

                if ranked.len() > 1 {
                    let (loser_id, loser_reward) = ranked.last().unwrap().clone();
                    let (winner_id, winner_reward) = ranked.first().unwrap().clone();
                    
                    println!("❌ Eliminating {} (Reward: {:.2}). Leader is {} (Reward: {:.2})", loser_id, loser_reward, winner_id, winner_reward);

                    if let Some(p) = parts.get_mut(&loser_id) {
                        p.status = RaceStatus::Eliminated;
                    }
                    
                    let active_count_after = ranked.len() - 1;
                    
                    if active_count_after == 1 {
                        println!("🏆 Finalist determined: {}! Respawning everyone as clones for the Victory Lap!", winner_id);
                        
                        let winner_path = self.best_model_path.lock().unwrap().clone();
                        let winner_settings = self.best_model_settings.lock().unwrap().clone();
                        
                        // Respawn Phase: Reactivate everyone
                        let mut active_guard = self.active_participants.lock().unwrap();
                        let mut all_ids = Vec::new();
                        for (pid, p) in parts.iter_mut() {
                            p.status = RaceStatus::Running; 
                            all_ids.push(pid.clone());
                        }
                        *active_guard = all_ids; 
                        
                        *next_ckpt_idx += 1;
                        self.barrier.notify_all();
                        
                        if let (Some(path), Some(set)) = (winner_path, winner_settings) {
                            return (true, Some((path, set)));
                        } else {
                            return (true, None);
                        }
                    } else {
                        // Eliminate loser from active set
                        let mut active_guard = self.active_participants.lock().unwrap();
                        active_guard.retain(|x| x != &loser_id);
                    }
                }

                *next_ckpt_idx += 1;
                self.barrier.notify_all();
            } else {
                let _unused = self.barrier.wait(parts).unwrap();
            }
        }
        
        let parts = self.participants.lock().unwrap();
        if let Some(p) = parts.get(id) {
            match p.status {
                RaceStatus::Eliminated => {
                    // Enter a wait loop until status changes (Respawn) or race ends
                    // We need to release the lock 'parts' to wait on condvar, but wait takes the lock.
                    // However, we are ALREADY holding 'parts' (MutexGuard).
                    // We need to enter a wait loop using the same barrier?
                    // Barrier notifies on Checkpoint.
                    // If we are eliminated, we wait for next checkpoint.
                    
                    // Actually, if we return false, the trainer exits.
                    // If we want to support "Respawn", we must NOT exit. We must wait.
                    // Let's force a wait here.
                    
                    println!("💤 Agent {} sleeping (Eliminated)...", id);
                    
                    // Problem: barrier.wait consumes the guard and returns it.
                    // We need to loop.
                    let mut current_parts = parts;
                    loop {
                        current_parts = self.barrier.wait(current_parts).unwrap();
                        
                        // Check if status changed to Running
                        if let Some(me) = current_parts.get(id) {
                            if me.status == RaceStatus::Running {
                                println!("⚡ Agent {} Respawned!", id);
                                // Check for pending reload data?
                                // Simplified: If we just woke up and are running, grab the winner data.
                                let winner_path = self.best_model_path.lock().unwrap().clone();
                                let winner_settings = self.best_model_settings.lock().unwrap().clone();
                                
                                if let (Some(path), Some(set)) = (winner_path, winner_settings) {
                                    return (true, Some((path, set)));
                                }
                                return (true, None);
                            }
                        }
                        
                        // Check if race ended (no more checkpoints)
                        let idx = self.next_checkpoint_idx.lock().unwrap();
                        if *idx >= self.checkpoints.len() {
                             return (false, None); // Race over
                        }
                    }
                },
                RaceStatus::Winner => (true, None),
                RaceStatus::Running | RaceStatus::WaitingForOthers => {
                    // If we just passed the final checkpoint (Respawn event), we might have data to load.
                    // But usually, only the one who triggered the checkpoint gets the immediate return.
                    // Others wake up from 'wait' above.
                    // If we fall through here, we are just running normal.
                    
                    // Check if we are in the "Respawn" moment?
                    // We can check if 'best_model_path' is set AND we haven't loaded it?
                    // To avoid infinite reloading, we rely on the fact that only the checkpoint event returns Some.
                    // Wait, the checkpoint event (all_arrived block) returns Some only for the caller.
                    // The others are waiting in `barrier.wait`.
                    // They wake up and go to this block.
                    // They return (true, None).
                    // THIS IS A BUG. They miss the reload!
                    
                    // Fix: We need to return the reload data for everyone at that specific moment.
                    // But how to synchronize?
                    // Maybe store "reload_command" in RaceParticipant?
                    
                    // Let's implement pending_reload in RaceParticipant.
                    // Or simpler:
                    // If active_count_after == 1 logic runs, it sets a global "broadcast_reload" flag?
                    
                    // Let's fix the logic for those waking up from wait.
                    // They are inside `else { barrier.wait }`.
                    // When they wake up, they continue to `match p.status`.
                    
                    // If we are in the Respawn phase, EVERYONE (Winner + Losers) continues.
                    // Losers are handled in the Eliminated block above (they wait until Running).
                    // Running agents (Winner or survivors of intermediate rounds) wake up here.
                    // If it was the final round, they might need reload? 
                    // Winner doesn't need to reload himself (optional).
                    
                    // So only the "Eliminated -> Running" path needs reload. I handled that in the loop above.
                    // Intermediate survivors?
                    // In the 25%->50% phase, survivors just continue. No reload.
                    // Only at 75%->100%, survivors (only 1 winner) continue.
                    // Wait, if N=4.
                    // 25%: 1 Eliminated. 3 Survivors. No reload.
                    // 50%: 1 Eliminated. 2 Survivors. No reload.
                    // 75%: 1 Eliminated. 1 Survivor (Winner). Winner continues. 
                    //      AND 3 Eliminated get respawned.
                    // So the Winner falls here. Returns (true, None). Correct.
                    // The Eliminated fall in the loop above. When status becomes Running, they return (true, Some(...)). Correct.
                    
                    (true, None)
                }
            }
        } else {
            (false, None)
        }
    }
}
