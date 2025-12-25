use std::sync::{Arc, atomic::AtomicBool};
use std::thread;
use std::collections::HashMap;
use crate::trainer::{Trainer, settings::TrainerSettings, GuiUpdate, ChampionTracker, race::RaceController};
use crate::agent::AgentType;
use crate::pages::config_types::{AlgoConfig, CheckpointMode};

use std::sync::mpsc::{channel, Sender, Receiver};

// Handler for spawning Trainer threads (renamed logic, but struct name kept for compatibility if desired, 
// though refactoring name is better. Let's keep TcpServerHandler but logic is Trainer)
pub struct TcpServerHandler {
    pub trainer_handles: Vec<thread::JoinHandle<()>>,
    pub shutdown_requested: Arc<AtomicBool>,
    pub gui_rx: Option<Receiver<GuiUpdate>>,
    gui_tx: Sender<GuiUpdate>,
    pub champion_tracker: Arc<ChampionTracker>,
}

impl TcpServerHandler {
    pub fn new(shutdown_flag: Arc<AtomicBool>) -> Self {
        let (tx, rx) = channel();
        Self {
            trainer_handles: Vec::new(),
            shutdown_requested: shutdown_flag,
            gui_rx: Some(rx),
            gui_tx: tx,
            champion_tracker: Arc::new(ChampionTracker::new()), 
        }
    }

    pub fn start_tcp_servers(
        &mut self, 
        launched_ports: &[u16], 
        port_algorithm_map: &HashMap<u16, String>,
        algo_configs: &HashMap<String, AlgoConfig>,
        output_path: &str,
        // Global Hyperparams from HomePage
        learning_rate: f32,
        hidden_units: u32,
        max_steps: u64,
        summary_freq: u32,
        reward_gamma: f32,
        checkpoint_interval: u32,
        keep_checkpoints: u32,
        init_path: &str,
        checkpoint_mode: CheckpointMode,
        device: &str,
        enable_race_mode: bool, // Added
    ) {
        // Reset champion tracker for a new run
        if let Ok(mut best) = self.champion_tracker.current_best.lock() {
            *best = None;
        }

        // 1. Group ports by Algorithm
        // If Race Mode is ON, we force 1 port per group (Unique groups)
        let mut algo_groups: HashMap<String, Vec<String>> = HashMap::new();

        for (i, &port) in launched_ports.iter().enumerate() {
            let algo_name = port_algorithm_map.get(&port)
                .cloned()
                .unwrap_or_else(|| "sac".to_string());
            
            let group_key = if enable_race_mode {
                format!("{}_{}", algo_name, i) // Unique key per env for Race
            } else {
                algo_name.clone() // Shared key for Standard
            };
            
            // Note: If race mode, we might want to preserve the 'algo_name' for config lookup later.
            // But we store the group key. We'll handle config lookup carefully.
            
            algo_groups.entry(group_key)
                .or_insert_with(Vec::new)
                .push(port.to_string());
        }

        // Initialize Race Controller if needed
        let race_controller = if enable_race_mode {
            let participant_ids: Vec<String> = algo_groups.values().flatten().cloned().collect();
            println!("🏁 Initializing Race Controller for {} participants. Max Steps: {}", participant_ids.len(), max_steps);
            Some(Arc::new(RaceController::new(max_steps as usize, participant_ids)))
        } else {
            None
        };

        println!("🏗️ Distributed Training Grouping: {:?}", algo_groups);

        // 2. Spawn ONE Trainer per Algorithm Group
        for (group_key, channel_ids) in algo_groups {
            // Extract real algo name from group key (e.g. "PPO_0" -> "PPO") if race mode
            let algo_name = if enable_race_mode {
                // Split by last underscore? Or assume strict format?
                // Simpler: Use the port map again for the first channel in the group
                let first_port = channel_ids[0].parse::<u16>().unwrap_or(0);
                port_algorithm_map.get(&first_port).cloned().unwrap_or("sac".to_string())
            } else {
                group_key.clone()
            };

            let is_dummy = algo_name == "DUMMY";

            // Get config
            let config = if is_dummy {
                AlgoConfig::ppo() // Dummy config
            } else if let Some(cfg) = algo_configs.get(&algo_name) {
                cfg.clone()
            } else {
                eprintln!("⚠️ No config found for algorithm '{}', using default SAC.", algo_name);
                AlgoConfig::sac()
            };

            // Map AlgoConfig -> TrainerSettings
            // CRITICAL: For Race Mode, each trainer needs a unique output path to avoid overwriting checkpoints!
            // Standard: results/run_id/checkpoint/algo_name
            // Race: results/run_id/checkpoint/algo_name_port
            let effective_output_path = if enable_race_mode {
                 format!("{}/race_{}", output_path, channel_ids[0])
            } else {
                 output_path.to_string()
            };

            let settings = map_config(
                &algo_name, 
                &config, 
                &effective_output_path, 
                learning_rate, 
                hidden_units, 
                max_steps, 
                summary_freq, 
                reward_gamma,
                checkpoint_interval,
                keep_checkpoints,
                init_path,
                checkpoint_mode,
                device,
            );
            
            let tx_clone = self.gui_tx.clone();
            let shutdown_clone = self.shutdown_requested.clone(); 
            let tracker_clone = self.champion_tracker.clone();
            let race_ctrl_clone = race_controller.clone();

            println!("🚀 Spawning {} for: {} (Group: {}) | Managing Envs: {:?}", 
                if is_dummy { "VISUAL DUMMY" } else { "CENTRAL Trainer" }, 
                algo_name, 
                group_key,
                channel_ids
            );

            // Spawn Thread
            let handle = thread::spawn(move || {
                // Initialize Trainer with LIST of channels
                let mut trainer = Trainer::new(
                    settings, 
                    Some(tx_clone), 
                    channel_ids.clone(), 
                    Some(tracker_clone), 
                    Some(shutdown_clone),
                    race_ctrl_clone // Pass controller
                );
                
                // Run
                let result = if is_dummy {
                    trainer.run_dummy()
                } else {
                    trainer.run()
                };

                if let Err(e) = result {
                     eprintln!("❌ Trainer [Group:{}] crashed: {}", group_key, e);
                } else {
                     println!("✅ Trainer [Group:{}] finished gracefully.", group_key);
                }
            });

            self.trainer_handles.push(handle);
        }
    }

    pub fn stop_servers(&mut self) {
        self.trainer_handles.clear();
        println!("🛑 Trainer handles cleared. Threads will exit when logic allows.");
    }
}

fn map_config(
    algo_name: &str, 
    cfg: &AlgoConfig, 
    output_path: &str,
    lr: f32,
    hu: u32,
    ms: u64,
    sf: u32,
    gamma: f32,
    ci: u32,
    kc: u32,
    ip: &str,
    mode: CheckpointMode,
    device: &str,
) -> TrainerSettings {
    let algorithm = match algo_name.to_lowercase().as_str() {
        "ppo" => AgentType::PPO,
        "ppo_et" => AgentType::PPO_ET,
        "ppo_ce" => AgentType::PPO_CE,
        "sac" | "drqv2" => AgentType::SAC, 
        "tqc" => AgentType::TQC,
        "crossq" => AgentType::CrossQ,
        "bc" => AgentType::BC,
        "poca" | "mappo" => AgentType::POCA,
        "tdsac" => AgentType::TDSAC,
        "td3" => AgentType::TD3,
        _ => AgentType::SAC,
    };

    TrainerSettings {
        algorithm,
        batch_size: cfg.batch_size as usize,
        buffer_size: cfg.buffer_size as usize,
        learning_rate: lr,
        hidden_units: hu as usize,
        summary_freq: sf as usize,
        checkpoint_interval: ci as usize,
        keep_checkpoints: kc as usize,
        max_steps: ms as usize,
        resume: mode == CheckpointMode::Resume || !ip.is_empty(), 
        epsilon: cfg.epsilon,
        tau: Some(cfg.tau),
        gamma: gamma,
        num_epochs: cfg.num_epoch.unwrap_or(3) as usize,
        lambd: cfg.lambd,
        entropy_coef: cfg.init_entcoef,
        policy_delay: cfg.policy_delay.map(|x| x as usize),
        n_quantiles: cfg.n_quantiles.map(|x| x as usize),
        n_to_drop: cfg.n_to_drop.map(|x| x as usize),
        curiosity_strength: cfg.curiosity_strength,
        curiosity_learning_rate: cfg.curiosity_learning_rate,
        show_obs: false, 
        output_path: output_path.to_string(),
        init_path: ip.to_string(),
        memory_size: cfg.memory_size.map(|x| x as usize),
        sequence_length: cfg.memory_sequence_length.map(|x| x as usize),
        device: device.to_string(),
    }
}
