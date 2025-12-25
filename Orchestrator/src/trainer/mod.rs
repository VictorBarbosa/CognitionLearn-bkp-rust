pub mod settings;
pub mod race; // Added

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::str;
use crate::agent::{RLAgent, AgentType};
use crate::trainer::settings::TrainerSettings;
use crate::trainer::race::RaceController; // Added
use crate::channel::CommunicationChannel;
use crate::protocol::{self, AgentAction, AgentInfo};
use crate::sac;
use crate::td3;
use crate::ppo;
use crate::tdsac;
use crate::tqc;
use crate::crossq;
use crate::bc;
use crate::poca;


use tensorboard_rs::summary_writer::SummaryWriter;
use tch::Device;
use serde::{Serialize, Deserialize};
use std::fs;

use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};

#[derive(Serialize, Deserialize, Debug)]
pub struct TrainerMetadata {
    pub total_train_steps: usize,
    pub total_transitions: usize,
    pub total_steps: usize,
    pub last_avg_reward: f32,
}

#[derive(Debug, Clone)]
pub struct ChampionModel {
    pub milestone: usize,
    pub reward: f32,
    pub path: String,
    pub algorithm: AgentType,
    pub port: String, // Added
}

pub struct ChampionTracker {
    pub current_best: Mutex<Option<ChampionModel>>,
}

impl ChampionTracker {
    pub fn new() -> Self {
        Self { current_best: Mutex::new(None) }
    }

    pub fn report_performance(&self, milestone: usize, reward: f32, path: &str, algorithm: AgentType, port: &str) -> bool {
        let mut best = self.current_best.lock().unwrap();
        
        let update_needed = match &*best {
            None => true,
            Some(curr) => {
                if milestone > curr.milestone {
                    true 
                } else if milestone == curr.milestone && reward > curr.reward {
                    true 
                } else {
                    false
                }
            }
        };

        if update_needed {
            println!("🏆 NEW CHAMPION: {:?} from Port {} at Milestone {} with Reward {:.4}", algorithm, port, milestone, reward);
            *best = Some(ChampionModel {
                milestone,
                reward,
                path: path.to_string(),
                algorithm,
                port: port.to_string(),
            });
        }
        update_needed
    }
}

// #[derive(Debug, Clone)]
pub enum GuiUpdate {
    Log(String),
    StepInfo {
        step: usize,
        avg_reward: f32,
        buffer_size: usize,
        behavior_name: String,
    },
    TrainingUpdate {
        step: usize,
        metrics: std::collections::HashMap<String, f32>,
        behavior_name: String,
    },
    RaceConfig {
        total_steps: usize,
        checkpoints: Vec<f32>,
    },
}

pub struct Trainer {
    pub settings: TrainerSettings,
    pub writer: SummaryWriter,
    pub agent: Option<Box<dyn RLAgent>>,
    pub total_train_steps: usize,
    pub total_transitions: usize,
    pub total_steps: usize,
    pub training_steps_credit: usize,
    pub agent_cumulative_rewards: HashMap<(String, i32), f32>,
    pub agent_episode_lengths: HashMap<(String, i32), usize>,
    pub last_avg_reward: f32,
    pub agent_last_obs: HashMap<(String, i32), Vec<f32>>,
    pub agent_last_action: HashMap<(String, i32), Vec<f32>>,
    pub device: Device,
    pub gui_sender: Option<Sender<GuiUpdate>>,
    pub channel_ids: Vec<String>,
    pub champion_tracker: Option<Arc<ChampionTracker>>, // Added
    pub race_controller: Option<Arc<RaceController>>, // Added
    pub shutdown_signal: Option<Arc<AtomicBool>>,
    pub last_checkpoint_step: usize,
    pub sensor_shapes: Option<Vec<i64>>, // Added
}

impl Trainer {
    pub fn new(
        settings: TrainerSettings, 
        gui_sender: Option<Sender<GuiUpdate>>, 
        channel_ids: Vec<String>,
        champion_tracker: Option<Arc<ChampionTracker>>, // Added
        shutdown_signal: Option<Arc<AtomicBool>>,
        race_controller: Option<Arc<RaceController>>, // Added
    ) -> Self {
// Trainer::new logic
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        // Use output_path for logs
        let log_dir = format!("{}/logs/run_{}_{:?}_{:?}", settings.output_path, timestamp, settings.algorithm, channel_ids);
        let writer = SummaryWriter::new(&log_dir);
        
        let device = match settings.device.to_lowercase().as_str() {
            "cuda" => {
                println!("🚀 Using CUDA GPU backend");
                Device::Cuda(0)
            },
            "mps" => {
                println!("🚀 Using Metal GPU backend (MPS)");
                Device::Mps
            },
            _ => {
                println!("⚠️ Using CPU backend");
                Device::Cpu
            }
        };
        
        let trainer = Self {
            settings,
            writer,
            agent: None,
            total_train_steps: 0,
            total_transitions: 0,
            total_steps: 0,
            training_steps_credit: 0,
            agent_cumulative_rewards: HashMap::new(),
            agent_episode_lengths: HashMap::new(),
            last_avg_reward: 0.0,
            agent_last_obs: HashMap::new(),
            agent_last_action: HashMap::new(),
            device,
            gui_sender,
            channel_ids,
            champion_tracker, // Added
            shutdown_signal,
            last_checkpoint_step: 0,
            race_controller, // Added
            sensor_shapes: None,
        };
        
        // ...
        trainer
    }

    pub fn run_dummy(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🎥 Orchestrator DUMMY (Visual Progress) Started.");
        
        let channel_id = &self.channel_ids[0]; 
        
        // 1. Perform Handshake for Dummy Environment
        let handshake_id = format!("{}_handshake", channel_id);
        let mut handshake_channel = CommunicationChannel::create(&handshake_id, 4096)?;
        
        println!("⏳ Waiting for DUMMY handshake on {}...", handshake_id);
        handshake_channel.listen(|_data| {
            println!("🤝 DUMMY Handshake Received.");
            Ok("seed:12345\ncommunicationVersion:1.5.0\n".as_bytes().to_vec())
        }, 60000)?; // 60s timeout
        
        // 2. Open Main Channel
        let mut main_channel = CommunicationChannel::create(channel_id, 10_485_760)?;
        println!("✅ DUMMY Handshake complete. Main channel active.");

        let mut current_champion_path = String::new();
        let mut local_obs_dim = 0;

        loop {
            // Check shutdown signal
            if let Some(signal) = &self.shutdown_signal {
                if signal.load(Ordering::Relaxed) { break Ok(()); }
            }

            // Check for new champion
            let mut new_champion = None;
            if let Some(tracker) = &self.champion_tracker {
                let best = tracker.current_best.lock().unwrap();
                if let Some(champion) = &*best {
                    if champion.path != current_champion_path {
                        new_champion = Some(champion.clone());
                    }
                }
            }

            // If we have a new champion AND we know the observation dimension
            if let Some(champion) = new_champion {
                if local_obs_dim > 0 {
                    println!("🔄 Dummy loading new champion: {} (Milestone: {})", champion.path, champion.milestone);
                    current_champion_path = champion.path.clone();
                    
                    // Use a dummy AgentInfo with the CORRECT discovered dimension
                    let dummy_info = AgentInfo { 
                        id: 0, observations: vec![0.0; local_obs_dim],
                        reward: 0.0, done: false, max_step_reached: false, sensor_shapes: vec![]
                    };
                    
                    let original_algo = self.settings.algorithm;
                    self.settings.algorithm = champion.algorithm;
                    self.init_agent("dummy", &dummy_info);
                    self.settings.algorithm = original_algo;

                    if let Some(agent) = self.agent.as_mut() {
                        let _ = agent.load(&current_champion_path);
                    }
                }
            }

            if main_channel.has_msg() {
                let _ = main_channel.listen(|data| {
                    let received_str = std::str::from_utf8(&data).unwrap_or("");
                    if received_str.starts_with("STEP") {
                        if let Ok(all_agent_infos) = protocol::parse_step_data(received_str) {
                            let mut all_actions = HashMap::new();
                            for (behavior, infos) in all_agent_infos {
                                // Discover observation dimension from the first message
                                if local_obs_dim == 0 && !infos.is_empty() {
                                    local_obs_dim = infos[0].observations.len();
                                    println!("📡 DUMMY discovered obs_dim: {}", local_obs_dim);
                                }

                                let mut actions = Vec::new();
                                if let Some(agent) = self.agent.as_ref() {
                                    for info in infos {
                                        let act = agent.select_action(&info.observations, true);
                                        actions.push(AgentAction { continuous_actions: act });
                                    }
                                } else {
                                    // No model yet, send zero actions (acts as 'waiting' state)
                                    for _ in infos { actions.push(AgentAction { continuous_actions: vec![0.0; 2] }); }
                                }
                                all_actions.insert(behavior, actions);
                            }
                            
                            // Format action data and append metadata labels
                            let mut response = protocol::format_action_data(&all_actions);
                            
                            // Add Champion Metadata for Unity UI
                            if let Some(tracker) = &self.champion_tracker {
                                let best = tracker.current_best.lock().unwrap();
                                if let Some(champion) = &*best {
                                    response.push_str("\nLABELS\n");
                                    response.push_str(&format!("Algorithm:{:?}\n", champion.algorithm));
                                    response.push_str(&format!("Port:{}\n", champion.port));
                                    response.push_str(&format!("Step:{}\n", champion.milestone));
                                }
                            }
                            
                            return Ok(response.as_bytes().to_vec());
                        }
                    }
                    Ok("ACK".as_bytes().to_vec())
                }, 100);
            } else {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Orchestrator Trainer Started (Distributed Mode).");
        println!("📡 Managing {} channels: {:?}", self.channel_ids.len(), self.channel_ids);

        if self.settings.resume {
            println!("🔄 Mode: RESUME");
        } else {
            println!("🆕 Mode: FRESH START");
        }

        const CHANNEL_SIZE: u64 = 10_485_760; // 10MB
        const TIMEOUT_MS: u64 = 60000; // 60s handshake timeout

        // Initialize all channels first to ensure files exist and we are ready to listen
        let mut channels = Vec::new(); 
        for id in &self.channel_ids {
            let handshake_id = format!("{}_handshake", id);
            let main_id = id.clone();
            
            // Create channels immediately
            let handshake_channel = CommunicationChannel::create(&handshake_id, 4096)?;
            let main_channel = CommunicationChannel::create(&main_id, CHANNEL_SIZE)?;
            
            // Push to pending list with state
            // (id, main_channel, handshake_channel, is_ready)
            channels.push((id.clone(), main_channel, Some(handshake_channel), false));
        }

        println!("⏳ Waiting for handshakes from {} environments...", channels.len());
        
        // Parallel Handshake Loop
        let handshake_start_time = SystemTime::now();
        let mut connected_count = 0;
        
        while connected_count < channels.len() {
            // Check global timeout
            if let Ok(elapsed) = handshake_start_time.elapsed() {
                if elapsed.as_millis() as u64 > TIMEOUT_MS {
                    return Err(format!("❌ Global Handshake Timeout after {}ms. Connected {}/{}", TIMEOUT_MS, connected_count, channels.len()).into());
                }
            }

             let mut any_activity = false;

            // Poll all pending channels
            for (id, _, handshake_channel_opt, is_ready) in channels.iter_mut() {
                if *is_ready {
                    continue;
                }

                if let Some(handshake_channel) = handshake_channel_opt {
                     if handshake_channel.has_msg() {
                        any_activity = true;
                        // Process handshake immediately
                        handshake_channel.listen(|_data| {
                            println!("🤝 Received Handshake from {}.", id);
                            // TODO: In the future, parse _data for env capabilities
                            Ok("seed:12345\ncommunicationVersion:1.5.0\n".as_bytes().to_vec())
                        }, 1000)?; // Short timeout for read/write since we know msg exists
                        
                        // Mark as ready and drop handshake channel logic if needed (or keep it open? logic says drop/replace)
                        // Actually we need to keep Main Channel. Handshake channel can be dropped or ignored.
                        // For now we just mark ready.
                         println!("✅ Handshake complete for {}.", id);
                        *is_ready = true;
                        connected_count += 1;
                     }
                }
            }

            if !any_activity {
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        println!("🎉 All {} environments connected!", connected_count);

        // Cleanup: We don't need handshake channels anymore, just keep main channels
        // We will reconstruct the 'channels' vec expected by the main loop
        // The main loop expects: Vec<(String, CommunicationChannel)>
        let mut main_loop_channels: Vec<(String, CommunicationChannel)> = Vec::new();
        for (id, main, _, _) in channels.drain(..) {
             main_loop_channels.push((id, main));
        }
        
        let mut channels = main_loop_channels; // Shadow with correct type for main loop

        // Send Race Configuration to GUI if applicable
        if let Some(rc) = &self.race_controller {
            if let Some(tx) = &self.gui_sender {
                let _ = tx.send(GuiUpdate::RaceConfig { 
                    total_steps: rc.total_steps, 
                    checkpoints: rc.checkpoints.clone() 
                });
            }
        }

        let mut last_msg_time = SystemTime::now();
        const SILENCE_TIMEOUT: Duration = Duration::from_secs(60);

        loop {
            // Check shutdown signal
            if let Some(signal) = &self.shutdown_signal {
                if signal.load(Ordering::Relaxed) {
                    println!("🛑 Shutdown signal received. Stopping trainer loop.");
                    self.save_checkpoint();
                    break Ok(());
                }
            }

            // Check for GLOBAL silence timeout
            if let Ok(elapsed) = last_msg_time.elapsed() {
                if elapsed > SILENCE_TIMEOUT {
                    println!("⚠️ Connection lost (no heartbeats from ANY env for {:?}). Saving checkpoint and exiting.", elapsed);
                    self.save_checkpoint();
                    break Ok(());
                }
            }
            
            let mut any_activity_this_loop = false;
            let mut added_this_batch = 0;

            // Poll all channels
            for (env_index, (channel_id, main_channel)) in channels.iter_mut().enumerate() {
                // Generate a unique ID offset for this environment to prevent Agent ID collisions in the Replay Buffer
                // Assuming max 100,000 agents per env (safe).
                let env_id_offset = (env_index as i32) * 100_000;
                
                if main_channel.has_msg() {
                    any_activity_this_loop = true;
                    //println!("📨 Processing message from {}", channel_id); // Debug verbose
                    
                    let mut msg_type = String::new();

                    // Timeout 10ms just to be safe if has_msg was true but read blocks slightly (shouldn't happen often)
                    let res = main_channel.listen(|data: Vec<u8>| {
                        let received_str = str::from_utf8(&data).unwrap_or("");
                        let first_line = received_str.lines().next().map(|s| s.trim()).unwrap_or("");
                        let first_line_upper = first_line.to_uppercase();
                        msg_type = first_line_upper.clone();
                        
                        if msg_type == "HEARTBEAT" {
                            Ok("ACK".as_bytes().to_vec())
                        } else if msg_type == "STEP" {
                             match protocol::parse_step_data(&received_str) {
                                Ok(all_agent_infos) => {
                                    self.total_steps += 1;
                                    let num_behaviors = all_agent_infos.len();

                                    let mut all_actions = HashMap::new();
                                    
                                    for (behavior_name, agent_infos) in all_agent_infos {
                                        let mut agent_actions = Vec::new();
                                        
                                        if self.agent.is_none() && !agent_infos.is_empty() {
                                            self.init_agent(&behavior_name, &agent_infos[0]);
                                        }

                                        if let Some(cur_agent) = self.agent.as_mut() {
                                            for a in agent_infos {
                                                let obs_len = a.observations.len();
                                                let expected_dim = cur_agent.get_obs_dim();
                                                
                                                // Unique composite key for this environment + agent combination
                                                let state_key = (channel_id.clone(), a.id);
                                                
                                                // 1. Record transition using composite key
                                                if let (Some(prev_obs), Some(prev_act)) = (self.agent_last_obs.get(&state_key), self.agent_last_action.get(&state_key)) {
                                                    if prev_obs.len() == expected_dim && a.observations.len() == expected_dim {
                                                        cur_agent.record_transition(
                                                            a.id + env_id_offset, // Global Unique ID
                                                            prev_obs.clone(),
                                                            prev_act.clone(),
                                                            a.reward,
                                                            a.observations.clone(),
                                                            a.done,
                                                        );
                                                        self.total_transitions += 1;
                                                        added_this_batch += 1;
                                                        
                                                        let cum_reward = self.agent_cumulative_rewards.entry(state_key.clone()).or_insert(0.0);
                                                        *cum_reward += a.reward;
                                                        let ep_len = self.agent_episode_lengths.entry(state_key.clone()).or_insert(0);
                                                        *ep_len += 1;
                                                        
                                                        if a.done {
                                                            self.writer.add_scalar("Environment/Cumulative Reward", *cum_reward, self.total_transitions);
                                                            self.writer.add_scalar("Environment/Episode Length", *ep_len as f32, self.total_transitions);
                                                            self.writer.flush();
                                                            *cum_reward = 0.0;
                                                            *ep_len = 0;
                                                        }
                                                    }
                                                }

                                                if self.settings.show_obs {
                                                    println!("👀 Env {} Agent {} Obs: {:?}", channel_id, a.id, a.observations);
                                                }
                                                
                                                let action = if obs_len == expected_dim {
                                                    let act = cur_agent.select_action(&a.observations, false);
                                                    if a.done {
                                                        self.agent_last_obs.remove(&state_key);
                                                        self.agent_last_action.remove(&state_key);
                                                    } else {
                                                        self.agent_last_obs.insert(state_key.clone(), a.observations.clone());
                                                        self.agent_last_action.insert(state_key.clone(), act.clone());
                                                    }
                                                    act
                                                } else {
                                                    if obs_len > 0 {
                                                        eprintln!("⚠️ Warning: Observation dim mismatch! Expected {}, got {}. Sending zero action.", expected_dim, obs_len);
                                                    }
                                                    vec![0.0; cur_agent.get_act_dim()]
                                                };
                                                agent_actions.push(AgentAction { continuous_actions: action });
                                            }
                                        }
                                        all_actions.insert(behavior_name, agent_actions);
                                    }
                                    
                                    // Progress log every 100 steps (GLOBAL steps)
                                    if self.total_steps % 100 == 0 {
                                        if let Some(cur_agent) = self.agent.as_ref() {
                                            let buffer_size = cur_agent.get_buffer_size();
                                            
                                            // Calculate stats PER ENVIRONMENT
                                            // We use the 'channel_id' (Env ID) to filter rewards
                                            let this_env_rewards: Vec<f32> = self.agent_cumulative_rewards
                                                .iter()
                                                .filter(|(k, _)| k.0 == *channel_id)
                                                .map(|(_, v)| *v)
                                                .collect();
                                            
                                            let count = this_env_rewards.len().max(1) as f32;
                                            let sum_rewards: f32 = this_env_rewards.iter().sum();
                                            let current_avg = sum_rewards / count;
                                            
                                            // Simple comparison (could be improved to store last per-env reward)
                                            let arrow = if current_avg > self.last_avg_reward { "⬆️" } else { "➡️" }; 

                                            let log_prefix = format!("[Env {}]", channel_id);
                                            println!("👣 {} STEP #{} | Behaviors: {} | Buffer: {}/{} | Reward: {:.4} {}", 
                                                log_prefix, self.total_steps, num_behaviors, buffer_size, self.settings.buffer_size, current_avg, arrow);
                                            
                                            if let Some(ref tx) = self.gui_sender {
                                                // Send explicit "Env X" label so MonitorPage can group them
                                                let _ = tx.send(GuiUpdate::StepInfo {
                                                    step: self.total_steps,
                                                    avg_reward: current_avg,
                                                    buffer_size,
                                                    behavior_name: format!("Env {} - {:?}", channel_id, self.settings.algorithm),
                                                });
                                            }
                                            
                                            // We validly update global last_avg just for the arrow logic next time (apprx)
                                            self.last_avg_reward = current_avg;
                                        }
                                    }
                                    Ok(protocol::format_action_data(&all_actions).as_bytes().to_vec())
                                }
                                Err(e) => {
                                    eprintln!("❌ Error parsing STEP: {}", e);
                                    Ok("ERROR".as_bytes().to_vec())
                                }
                            }
                        } else if msg_type == "TRANSITION" {
                             match protocol::parse_transition_data(&received_str) {
                                Ok(all_transitions) => {
                                    for (behavior, transitions) in all_transitions {
                                        if self.agent.is_none() && !transitions.is_empty() {
                                             let dummy = AgentInfo { 
                                                id: transitions[0].id, 
                                                observations: transitions[0].observations.clone(),
                                                reward: 0.0, done: false, max_step_reached: false,
                                                sensor_shapes: transitions[0].sensor_shapes.clone(),
                                            };
                                            self.init_agent(&behavior, &dummy);
                                        }

                                        if let Some(cur_agent) = self.agent.as_mut() {
                                            for t in transitions {
                                                let state_key = (channel_id.clone(), t.id);
                                                
                                                cur_agent.record_transition(
                                                    t.id + env_id_offset, t.observations, t.actions, t.reward, t.next_observations, t.done,
                                                );
                                                added_this_batch += 1;
                                                self.total_transitions += 1;

                                                let cum_reward = self.agent_cumulative_rewards.entry(state_key.clone()).or_insert(0.0);
                                                *cum_reward += t.reward;
                                                let ep_len = self.agent_episode_lengths.entry(state_key.clone()).or_insert(0);
                                                *ep_len += 1;

                                                if t.done {
                                                    self.writer.add_scalar("Environment/Cumulative Reward", *cum_reward, self.total_transitions);
                                                    self.writer.add_scalar("Environment/Episode Length", *ep_len as f32, self.total_transitions);
                                                    self.writer.flush();
                                                    *cum_reward = 0.0;
                                                    *ep_len = 0;
                                                }
                                                
                                                if self.total_transitions % 100 == 0 {
                                                    let buffer_size = cur_agent.get_buffer_size();
                                                    let threshold = cur_agent.get_training_threshold();
                                                    let sum_rewards: f32 = self.agent_cumulative_rewards.values().sum();
                                                    let count = self.agent_cumulative_rewards.len().max(1) as f32;
                                                    let current_avg = sum_rewards / count;
                                                    let arrow = if current_avg > self.last_avg_reward { "⬆️" } else { "➡️" }; // Simplified
                                                    
                                                    let log_prefix = format!("[Env {}]", channel_id);
                                                    println!("👣 {} [{}] STEP #{} | Behaviors: ? | Buffer: {}/{} | Reward: {:.4} {}", 
                                                        log_prefix, behavior, self.total_transitions, buffer_size, threshold, current_avg, arrow);
                                                }
                                            }
                                        }
                                    }
                                    Ok("ACK".as_bytes().to_vec())
                                }
                                Err(e) => {
                                    eprintln!("❌ Error parsing TRANSITION: {}", e);
                                    Ok("ACK".as_bytes().to_vec())
                                }
                            }
                        } else {
                            Ok("ACK".as_bytes().to_vec())
                        }
                    }, 100); // Short timeout because we already checked has_msg

                    if res.is_ok() {
                        last_msg_time = SystemTime::now(); // Update timestamp on successful message
                    }
                }
            } // End of channel iteration

            // Check if we should sleep
            if !any_activity_this_loop {
                std::thread::sleep(Duration::from_millis(1));
            } else {
                 // If we had activity, update training steps
                 if added_this_batch > 0 {
                    self.training_steps_credit += added_this_batch;
                 }
                 
                 // Check training condition
                 let mut should_save = false;

                 if let Some(cur_agent) = self.agent.as_mut() {
                    let buffer_size = cur_agent.get_buffer_size();
                    let threshold = cur_agent.get_training_threshold();
                    
                    if self.training_steps_credit > 0 && buffer_size >= threshold {
                        let mut accumulated_metrics: HashMap<String, (f32, usize)> = HashMap::new();
                        let steps_to_run = std::cmp::min(self.training_steps_credit, 16); // Train max 16 steps per loop
                        
                        for _ in 0..steps_to_run {
                            if let Some(metrics) = cur_agent.train() {
                                self.total_train_steps += 1;
                                self.training_steps_credit -= 1;
                                for (name, value) in metrics {
                                    let entry = accumulated_metrics.entry(name).or_insert((0.0, 0));
                                    entry.0 += value;
                                    entry.1 += 1;
                                }

                                // Check for checkpoint based on ENVIRONMENT steps (total_steps or total_transitions)
                                // We use total_steps for consistency across algorithms
                                if self.total_steps >= self.last_checkpoint_step + self.settings.checkpoint_interval {
                                    should_save = true;
                                    self.last_checkpoint_step = self.total_steps;
                                }
                            }
                        }

                        // Summary
                        // Summary
                        let mut averaged_metrics = std::collections::HashMap::new();
                        for (name, (sum, count)) in accumulated_metrics {
                            let val = sum / count as f32;
                            self.writer.add_scalar(&name, val, self.total_train_steps);
                            averaged_metrics.insert(name, val);
                        }

                        if let Some(ref tx) = self.gui_sender {
                            let _ = tx.send(GuiUpdate::TrainingUpdate {
                                step: self.total_train_steps,
                                metrics: averaged_metrics,
                                behavior_name: format!("{:?}", self.settings.algorithm),
                            });
                        }
                    }
                }

                if should_save {
                    self.save_checkpoint();
                }

                // --- RACE LOGIC ---
                // Clone the Arc to avoid borrowing 'self' while we might need to mutate 'self.settings' later
                if let Some(rc) = self.race_controller.clone() {
                    let algo_str = format!("{:?}", self.settings.algorithm).to_lowercase();
                    let ckpt_path = format!("{}/checkpoint/{}/checkpoint.ot", self.settings.output_path, algo_str);
                    let my_id = self.channel_ids.first().cloned().unwrap_or("unknown".to_string());
                    
                    let (should_continue, load_info) = rc.report_progress(
                        &my_id, 
                        self.total_steps, 
                        self.last_avg_reward, 
                        self.settings.algorithm, 
                        &ckpt_path,
                        &self.settings
                    );
                    
                    if !should_continue {
                        println!("🛑 Race Controller ended training for agent {}.", my_id);
                        break Ok(()); 
                    }
                    
                    if let Some((path, new_settings)) = load_info {
                        println!("♻️ Race Controller ordered reload from: {}", path);
                        
                        // Check for Algorithm Swap
                        let algo_changed = new_settings.algorithm != self.settings.algorithm;
                        self.settings = new_settings; // Adopt winner's settings
                        
                        if algo_changed {
                             println!("🔄 Swapping Algorithm to {:?}", self.settings.algorithm);
                             if let Some(current_agent) = &self.agent {
                                 let obs_dim = current_agent.get_obs_dim();
                                 let act_dim = current_agent.get_act_dim();
                                 // Re-create agent with NEW settings
                                 self.agent = Some(self.create_agent(&self.settings, obs_dim, act_dim));
                             }
                        }

                        if let Some(agent) = self.agent.as_mut() {
                            if let Err(e) = agent.load(&path) {
                                eprintln!("❌ Failed to reload champion weights: {}", e);
                            } else {
                                println!("✅ Champion weights loaded successfully.");
                            }
                        }
                    }
                }
                // ------------------
            }
        }
    }

    fn init_agent(&mut self, behavior: &str, info: &AgentInfo) {
        let obs_dim = info.observations.len();
        let act_dim = 2; // Fixed for now, should be dynamic
        println!("🚀 Dynamic Init ({}): obs_dim={}, act_dim={}", behavior, obs_dim, act_dim);
        
        self.sensor_shapes = if info.sensor_shapes.is_empty() { None } else { Some(info.sensor_shapes.clone()) };

        self.agent = Some(self.create_agent(&self.settings, obs_dim, act_dim));

        if self.settings.resume || !self.settings.init_path.is_empty() {
            let algo_str = format!("{:?}", self.settings.algorithm).to_lowercase();
            
            // 1. Determine the path to load from
            let (ckpt_path, meta_path) = if !self.settings.init_path.is_empty() {
                let p = self.settings.init_path.clone();
                let m = p.replace(".ot", ".json").replace(".onnx", ".json");
                (p, m)
            } else {
                let base = format!("{}/checkpoint/{}", self.settings.output_path, algo_str);
                (format!("{}/checkpoint.ot", base), format!("{}/metadata.json", base))
            };

            // 2. Load Weights
            if std::path::Path::new(&ckpt_path).exists() {
                println!("💾 Loading Weights: {}", ckpt_path);
                if let Some(agent) = self.agent.as_mut() {
                    match agent.load(&ckpt_path) {
                        Ok(_) => println!("✅ Weights loaded successfully."),
                        Err(e) => eprintln!("❌ Failed to load weights: {}", e),
                    }
                }
            }

            // 3. Load Metadata (to resume steps/stats)
            if std::path::Path::new(&meta_path).exists() {
                if let Ok(content) = fs::read_to_string(&meta_path) {
                    if let Ok(meta) = serde_json::from_str::<TrainerMetadata>(&content) {
                        println!("📊 Resuming from Milestone: {} steps", meta.total_steps);
                        self.total_steps = meta.total_steps;
                        self.total_train_steps = meta.total_train_steps;
                        self.total_transitions = meta.total_transitions;
                        self.last_avg_reward = meta.last_avg_reward;
                        self.last_checkpoint_step = meta.total_steps;
                    }
                }
            }
        }
    }

    fn create_agent(&self, settings: &TrainerSettings, obs_dim: usize, act_dim: usize) -> Box<dyn RLAgent> {
        match settings.algorithm {
            AgentType::SAC => Box::new(sac::SAC::new(
                obs_dim, 
                act_dim, 
                settings.hidden_units, 
                settings.buffer_size, 
                settings.learning_rate as f64,
                settings.gamma as f64,
                settings.tau.unwrap_or(0.005) as f64,
                settings.entropy_coef.unwrap_or(0.2) as f64, // Alpha init
                self.device,
                self.sensor_shapes.clone(),
                settings.memory_size,
                settings.sequence_length
            )),
            AgentType::PPO | AgentType::PPO_ET | AgentType::PPO_CE => {
                let use_adaptive_entropy = matches!(settings.algorithm, AgentType::PPO_ET);
                let use_curiosity = matches!(settings.algorithm, AgentType::PPO_CE);
                
                Box::new(ppo::PPO::new(
                    obs_dim, 
                    act_dim, 
                    settings.hidden_units, 
                    settings.learning_rate as f64,
                    settings.gamma as f64,
                    settings.lambd.unwrap_or(0.95) as f64,
                    settings.epsilon.unwrap_or(0.2) as f64,
                    settings.entropy_coef.unwrap_or(0.01) as f64,
                    settings.num_epochs,
                    settings.buffer_size, // horizon
                    settings.batch_size,  // minibatch_size
                    self.device,
                    self.sensor_shapes.clone(),
                    use_adaptive_entropy,
                    use_curiosity,
                    settings.curiosity_strength.unwrap_or(0.01) as f64,
                    settings.curiosity_learning_rate.unwrap_or(3e-4) as f64,
                    settings.memory_size,
                    settings.sequence_length
                ))
            },
            AgentType::TD3 => Box::new(td3::TD3::new(
                obs_dim, 
                act_dim, 
                settings.hidden_units, 
                settings.buffer_size, 
                settings.learning_rate as f64,
                settings.gamma as f64,
                settings.tau.unwrap_or(0.005) as f64,
                settings.policy_delay.unwrap_or(2),
                self.device,
                self.sensor_shapes.clone(),
                settings.memory_size,
                settings.sequence_length
            )),
            AgentType::TDSAC => Box::new(tdsac::TDSAC::new(
                obs_dim, 
                act_dim, 
                settings.hidden_units, 
                settings.buffer_size, 
                settings.learning_rate as f64,
                settings.gamma as f64,
                settings.tau.unwrap_or(0.005) as f64,
                settings.entropy_coef.unwrap_or(0.2) as f64, // Alpha
                settings.policy_delay.unwrap_or(2),
                self.device,
                self.sensor_shapes.clone(),
                settings.memory_size,
                settings.sequence_length
            )),
            AgentType::TQC => Box::new(tqc::TQC::new(
                obs_dim, 
                act_dim, 
                settings.hidden_units, 
                settings.buffer_size, 
                settings.learning_rate as f64,
                settings.gamma as f64,
                settings.tau.unwrap_or(0.005) as f64,
                settings.entropy_coef.unwrap_or(0.2) as f64,
                settings.n_quantiles.unwrap_or(25),
                settings.n_to_drop.unwrap_or(2),
                self.device,
                self.sensor_shapes.clone(),
                settings.memory_size,
                settings.sequence_length
            )),
            AgentType::CrossQ => Box::new(crossq::CrossQ::new(
                obs_dim, 
                act_dim, 
                settings.hidden_units, 
                settings.buffer_size, 
                settings.learning_rate as f64,
                settings.gamma as f64,
                settings.entropy_coef.unwrap_or(0.2) as f64, // Alpha init
                self.device,
                self.sensor_shapes.clone(),
                settings.memory_size,
                settings.sequence_length
            )),
            AgentType::BC => Box::new(bc::BC::new(
                obs_dim, 
                act_dim, 
                settings.hidden_units, 
                settings.buffer_size, 
                settings.batch_size,
                settings.learning_rate as f64,
                self.device,
                self.sensor_shapes.clone(),
                settings.memory_size,
                settings.sequence_length
            )),
            AgentType::POCA => Box::new(poca::POCA::new(
                obs_dim, 
                act_dim, 
                settings.hidden_units, 
                settings.buffer_size, 
                settings.batch_size,
                settings.learning_rate as f64,
                settings.num_epochs,
                settings.gamma as f64,
                settings.lambd.unwrap_or(0.95) as f64,
                settings.epsilon.unwrap_or(0.2) as f64,
                settings.entropy_coef.unwrap_or(0.01) as f64,
                0.5, // vf_coef default
                self.device,
                self.sensor_shapes.clone(),
                settings.memory_size,
                settings.sequence_length
            )),
        }
    }

    fn save_checkpoint(&mut self) { 
        if let Some(cur_agent) = self.agent.as_ref() {
            let algo_str = format!("{:?}", self.settings.algorithm).to_lowercase();
            // Use output_path for checkpoints
            let ckpt_dir = format!("{}/checkpoint/{}", self.settings.output_path, algo_str);
            let ckpt_path = format!("{}/checkpoint.ot", ckpt_dir);
            let meta_path = format!("{}/metadata.json", ckpt_dir);
            
            if let Err(e) = std::fs::create_dir_all(&ckpt_dir) {
                eprintln!("❌ Error creating checkpoint directory: {}", e);
            }

            // Save Weights
            if let Err(e) = cur_agent.save(&ckpt_path) {
                eprintln!("❌ Error saving weights: {}", e);
            } else {
                println!("💾 Checkpoint saved: {} (#{})", ckpt_path, self.total_train_steps);
                
                // Save Metadata
                let meta = TrainerMetadata {
                    total_train_steps: self.total_train_steps,
                    total_transitions: self.total_transitions,
                    total_steps: self.total_steps,
                    last_avg_reward: self.last_avg_reward,
                };
                if let Ok(json) = serde_json::to_string_pretty(&meta) {
                    let _ = fs::write(meta_path, json);
                }

                // Report to Champion Tracker
                if let Some(tracker) = &self.champion_tracker {
                    // Milestone is the nearest lower multiple of checkpoint_interval
                    let milestone = (self.total_steps / self.settings.checkpoint_interval) * self.settings.checkpoint_interval;
                    let source_port = self.channel_ids.first().cloned().unwrap_or_else(|| "unknown".to_string());
                    tracker.report_performance(milestone, self.last_avg_reward, &ckpt_path, self.settings.algorithm, &source_port);
                }

                // Try to export to ONNX (PPO/SAC/TD3/TDSAC/TQC/CrossQ/BC/POCA + PPO Variants)
                if matches!(self.settings.algorithm, AgentType::PPO | AgentType::PPO_ET | AgentType::PPO_CE | AgentType::SAC | AgentType::TD3 | AgentType::TDSAC | AgentType::TQC | AgentType::CrossQ | AgentType::BC | AgentType::POCA) {
                    let onnx_path_latest = format!("{}/model.onnx", ckpt_dir);
                    let onnx_path_versioned = format!("{}/model-{}.onnx", ckpt_dir, self.total_steps);
                    
                    println!("🔄 Exporting to ONNX (Native Rust)...");
                    match cur_agent.export_onnx(&onnx_path_versioned) {
                        Ok(_) => {
                            println!("✅ ONNX Exported: {}", onnx_path_versioned);
                            // Also copy to latest for convenience
                            if let Err(e) = std::fs::copy(&onnx_path_versioned, &onnx_path_latest) {
                                eprintln!("⚠️ Failed to update latest model.onnx: {}", e);
                            } else {
                                println!("✅ Updated latest: {}", onnx_path_latest);
                            }

                            // --- FILE ROTATION LOGIC ---
                            if let Ok(entries) = std::fs::read_dir(&ckpt_dir) {
                                let mut versioned_models: Vec<(u64, std::path::PathBuf)> = entries
                                    .filter_map(|e| e.ok())
                                    .filter_map(|e| {
                                        let name = e.file_name().into_string().ok()?;
                                        if name.starts_with("model-") && name.ends_with(".onnx") {
                                            // Extract step number for sorting
                                            let step: u64 = name.replace("model-", "").replace(".onnx", "").parse().ok()?;
                                            Some((step, e.path()))
                                        } else { None }
                                    })
                                    .collect();

                                // Sort by step (ascending)
                                versioned_models.sort_by_key(|m| m.0);

                                // If we have more models than allowed, delete the oldest ones
                                if versioned_models.len() > self.settings.keep_checkpoints {
                                    let num_to_delete = versioned_models.len() - self.settings.keep_checkpoints;
                                    for i in 0..num_to_delete {
                                        let path_to_delete = &versioned_models[i].1;
                                        if let Err(e) = std::fs::remove_file(path_to_delete) {
                                            eprintln!("⚠️ Failed to rotate old model: {}", e);
                                        } else {
                                            println!("🧹 Rotated (deleted) old model: {:?}", path_to_delete.file_name());
                                        }
                                    }
                                }
                            }
                        },
                        Err(e) => eprintln!("⚠️ ONNX Export failed: {}", e),
                    }
                }
            }
        }
    }
}
