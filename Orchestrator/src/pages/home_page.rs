use eframe::egui;
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

// Import the new modules
use crate::pages::config_types::{ConfigSection, EngineSettings, CheckpointSettings, AlgorithmConfigStep, AlgorithmSelectionMode, AlgoConfig};
use crate::pages::settings;
use crate::pages::algorithms;
use crate::pages::unity_launcher::UnityLauncher;
use crate::pages::tcp_server::TcpServerHandler;
use crate::pages::monitor_page::MonitorPage;
use crate::protocol::{AgentAction, AgentInfo}; // Kept for potential future use or if I misread, but actually I should remove it if unused.
// Let's comment it out or remove.
// I will just remove the line.


pub struct HomePage {
    // Top-level settings
    pub total_env: u16,
    pub headless: bool,
    pub visual_progress: bool,
    pub device: String, // "cpu", "cuda", "mps"
    pub run_id: String,
    pub env_path: String, // To display the selected path
    pub results_path: String, // Path to save results (logs, models)
    pub shared_memory_path: String, // New field
    pub base_port: u16,

    // Global Hyperparameters
    pub learning_rate: f32,
    pub learning_rate_schedule: String,
    pub hidden_units: u32,
    pub num_layers: u32,
    pub normalize: bool,
    pub reward_gamma: f32,
    pub reward_strength: f32,
    pub max_steps: u64,
    pub summary_freq: u32,
    pub init_path: String,

    pub num_areas: u16,
    pub timeout_wait: u16,
    pub seed: i32,
    pub max_lifetime_restarts: u16,
    pub restarts_rate_limit_n: u16,
    pub restarts_rate_limit_period_s: u16,

    pub engine_settings: EngineSettings,
    pub checkpoint_settings: CheckpointSettings,
    pub current_config_section: ConfigSection, // Field to control the active section
    pub settings_open: bool, // To control whether the Settings accordion is open
    pub algorithms_open: bool, // To control whether the Algorithms accordion is open

    // Algorithms
    pub ppo_enabled: bool,
    pub sac_enabled: bool,
    pub td3_enabled: bool,
    pub tdsac_enabled: bool,
    pub tqc_enabled: bool,
    pub crossq_enabled: bool,
    pub drqv2_enabled: bool,
    pub ppo_et_enabled: bool,
    pub ppo_ce_enabled: bool,

    pub ppo_env_count: u32,
    pub sac_env_count: u32,
    pub td3_env_count: u32,
    pub tdsac_env_count: u32,
    pub tqc_env_count: u32,
    pub crossq_env_count: u32,
    pub drqv2_env_count: u32,
    pub ppo_et_env_count: u32,
    pub ppo_ce_env_count: u32,
    pub distributed_env_total: u32,

    pub checkpoint_interval: u32,  // Intervalo global de checkpoint
    pub keep_checkpoints: u32,    // Number of checkpoints to keep

    pub algorithm_configs: HashMap<String, AlgoConfig>,
    pub algorithm_selection_mode: AlgorithmSelectionMode, // New field for the selection mode
    pub algorithm_config_step: AlgorithmConfigStep,
    pub selected_algorithms_for_config: Vec<String>,
    pub current_config_algorithm_index: usize,
    
    // Unity launcher and TCP server
    pub unity_launcher: UnityLauncher,
    pub tcp_server_handler: TcpServerHandler,
    
    // Monitoring
    pub monitor_page: MonitorPage,
    pub show_monitor: bool,
}

impl HomePage {
    pub fn new() -> Self {
        let shutdown_requested = Arc::new(AtomicBool::new(false));
        
        let default_device = if cfg!(target_os = "macos") && tch::utils::has_mps() {
            "mps"
        } else if tch::Cuda::is_available() {
            "cuda"
        } else {
            "cpu"
        };

        Self {
            total_env: 1, 
            headless: false,
            visual_progress: false,
            device: default_device.to_string(),
            run_id: String::from(""),
            env_path: String::from(""),
            results_path: String::from(""), 
            shared_memory_path: std::env::temp_dir().join("cognition_memory").to_string_lossy().to_string(),
            base_port: 5005,

            // Global Hyperparameters Init
            learning_rate: 3e-4,
            learning_rate_schedule: String::from("linear"),
            hidden_units: 512,
            num_layers: 3,
            normalize: true,
            reward_gamma: 0.995,
            reward_strength: 1.0,
            max_steps: 500000000,
            summary_freq: 5000,
            init_path: String::from(""),

            num_areas: 1,
            timeout_wait: 60,
            seed: -1,
            max_lifetime_restarts: 10,
            restarts_rate_limit_n: 1,
            restarts_rate_limit_period_s: 60,

            engine_settings: EngineSettings::default(),
            checkpoint_settings: CheckpointSettings::default(),
            current_config_section: ConfigSection::Settings, 
            settings_open: true, 
            algorithms_open: false, 

            // Algorithms initialization
            ppo_enabled: false,
            sac_enabled: false,
            td3_enabled: true, 
            tdsac_enabled: false,
            tqc_enabled: false,
            crossq_enabled: false,
            drqv2_enabled: false,
            ppo_et_enabled: false,
            ppo_ce_enabled: false,

            ppo_env_count: 0,
            sac_env_count: 0,
            td3_env_count: 1, 
            tdsac_env_count: 0,
            tqc_env_count: 0,
            crossq_env_count: 0,
            drqv2_env_count: 0,
            ppo_et_env_count: 0,
            ppo_ce_env_count: 0,
            distributed_env_total: 1,
            checkpoint_interval: 5000,  
            keep_checkpoints: 5,       

            algorithm_selection_mode: AlgorithmSelectionMode::Same, 
            algorithm_config_step: AlgorithmConfigStep::Selection, 
            selected_algorithms_for_config: vec!["TD3".to_string()], 
            current_config_algorithm_index: 0,
            
            unity_launcher: UnityLauncher::new(),
            tcp_server_handler: TcpServerHandler::new(shutdown_requested),
            
            monitor_page: MonitorPage::new(),
            show_monitor: false,

            algorithm_configs: {
                let mut m = HashMap::new();
                m.insert("PPO".into(), AlgoConfig::ppo());
                m.insert("SAC".into(), AlgoConfig::sac());
                m.insert("TD3".into(), AlgoConfig::td3());
                m.insert("TDSAC".into(), AlgoConfig::tdsac());
                m.insert("TQC".into(), AlgoConfig::tqc());
                m.insert("CrossQ".into(), AlgoConfig::crossq());
                m.insert("DrQV2".into(), AlgoConfig::drqv2());
                m.insert("PPO_ET".into(), AlgoConfig::ppo_et());
                m.insert("PPO_CE".into(), AlgoConfig::ppo_ce());
                m
            },
        }
    }

    pub fn update(&mut self, ctx: &egui::Context) {
        ctx.set_visuals(egui::Visuals::dark()); 

        egui::TopBottomPanel::top("top_nav").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Orchestrator GUI");
                if self.unity_launcher.is_training_started {
                    ui.separator();
                    if ui.selectable_label(!self.show_monitor, "Configuration").clicked() {
                        self.show_monitor = false;
                    }
                    if ui.selectable_label(self.show_monitor, "Monitor").clicked() {
                        self.show_monitor = true;
                    }
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.show_monitor && self.unity_launcher.is_training_started {
                 // Force UI refresh to keep charts and logs updating automatically
                 ctx.request_repaint();

                 if let Some(rx) = &self.tcp_server_handler.gui_rx {
                     self.monitor_page.update(ui, rx);
                 } else {
                     ui.centered_and_justified(|ui| {
                         ui.label("Waiting for training backend to initialize...");
                     });
                 }
                 
                 ui.with_layout(egui::Layout::bottom_up(egui::Align::RIGHT), |ui| {
                      if ui.button(egui::RichText::new("Stop Training").color(egui::Color32::RED)).clicked() {
                           self.unity_launcher.stop_training();
                           self.tcp_server_handler.stop_servers();
                           self.show_monitor = false; 
                      }
                 });
                 
            } else {
                self.render_config_page(ui);
            }
        });
    }

    fn render_config_page(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            // Settings Accordion
            egui::CollapsingHeader::new(egui::RichText::new("Settings").size(20.0))
                .enabled(self.current_config_section == ConfigSection::Settings)
                .open(Some(self.settings_open))
                .show(ui, |ui| {
                    settings::render_settings_ui(
                        ui,
                        &mut self.total_env,
                        &mut self.headless,
                        &mut self.visual_progress,
                        &mut self.device,
                        &mut self.run_id,
                        &mut self.env_path,
                        &mut self.results_path,
                        &mut self.shared_memory_path,
                        &mut self.base_port,
                        &mut self.learning_rate,
                        &mut self.learning_rate_schedule,
                        &mut self.hidden_units,
                        &mut self.num_layers,
                        &mut self.normalize,
                        &mut self.reward_gamma,
                        &mut self.reward_strength,
                        &mut self.max_steps,
                        &mut self.summary_freq,
                        &mut self.init_path,
                        &mut self.num_areas,
                        &mut self.timeout_wait,
                        &mut self.seed,
                        &mut self.max_lifetime_restarts,
                        &mut self.restarts_rate_limit_n,
                        &mut self.restarts_rate_limit_period_s,
                        &mut self.engine_settings,
                        &mut self.checkpoint_settings,
                        &mut self.checkpoint_interval,
                        &mut self.keep_checkpoints,
                        &mut self.current_config_section,
                        &mut self.settings_open,
                        &mut self.algorithms_open,
                    );

                    if self.checkpoint_settings.mode == crate::pages::config_types::CheckpointMode::Resume {
                         ui.add_space(10.0);
                         ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui_res| {
                             if ui_res.button(egui::RichText::new("🔍 Auto-Detect Checkpoints & Resume").size(16.0).color(egui::Color32::LIGHT_BLUE)).clicked() {
                                 self.perform_auto_resume_discovery();
                             }
                         });
                    }
                });

            ui.add_space(10.0);

            // Algorithms Accordion
            egui::CollapsingHeader::new(egui::RichText::new("Algorithms").size(20.0))
                .enabled(self.current_config_section == ConfigSection::Algorithms)
                .open(Some(self.algorithms_open))
                .show(ui, |ui| {
                    algorithms::render_algorithm_selection_ui(
                        ui,
                        self.total_env,
                        &mut self.algorithm_selection_mode,
                        &mut self.ppo_enabled,
                        &mut self.sac_enabled,
                        &mut self.td3_enabled,
                        &mut self.tdsac_enabled,
                        &mut self.tqc_enabled,
                        &mut self.crossq_enabled,
                        &mut self.drqv2_enabled,
                        &mut self.ppo_et_enabled,
                        &mut self.ppo_ce_enabled,
                        &mut self.ppo_env_count,
                        &mut self.sac_env_count,
                        &mut self.td3_env_count,
                        &mut self.tdsac_env_count,
                        &mut self.tqc_env_count,
                        &mut self.crossq_env_count,
                        &mut self.drqv2_env_count,
                        &mut self.ppo_et_env_count,
                        &mut self.ppo_ce_env_count,
                        &mut self.distributed_env_total,
                        &mut self.algorithm_config_step,
                        &mut self.current_config_section,
                        &mut self.settings_open,
                        &mut self.algorithms_open,
                        &mut self.selected_algorithms_for_config,
                        &mut self.current_config_algorithm_index,
                        &mut self.algorithm_configs,
                    );
                    
                    if self.algorithm_config_step == AlgorithmConfigStep::Configuration && 
                       self.current_config_algorithm_index >= self.selected_algorithms_for_config.len().saturating_sub(1) {
                        ui.add_space(20.0);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui_buttons| {
                            if ui_buttons.add_enabled(!self.unity_launcher.is_training_started, egui::Button::new(egui::RichText::new("Start Train").size(18.0))).clicked() {
                                if self.results_path.trim().is_empty() {
                                    rfd::MessageDialog::new().set_title("Error").set_description("Results Path is empty").show();
                                    return;
                                }

                                self.unity_launcher.is_training_started = true;
                                self.tcp_server_handler.shutdown_requested.store(false, Ordering::SeqCst);
                                
                                // Pre-initialize Race Monitor if Race Mode is enabled
                                if self.checkpoint_settings.enable_race_mode {
                                    self.monitor_page.set_race_config(self.max_steps as usize, self.total_env as usize);
                                }

                                match self.unity_launcher.launch_unity_environment(
                                    &self.env_path,
                                    self.total_env,
                                    self.base_port,
                                    &self.selected_algorithms_for_config,
                                    self.num_areas,
                                    self.timeout_wait,
                                    self.seed,
                                    self.max_lifetime_restarts,
                                    self.restarts_rate_limit_n,
                                    self.restarts_rate_limit_period_s,
                                    self.headless,
                                    self.visual_progress,
                                    &self.engine_settings,
                                    &self.run_id,
                                    &self.device,
                                    &self.algorithm_configs,
                                    self.checkpoint_interval,
                                    self.keep_checkpoints,
                                    &self.results_path,
                                    &self.shared_memory_path, // Added
                                ) {
                                    Ok(_) => {
                                        if !self.unity_launcher.unity_processes.is_empty() {
                                            self.tcp_server_handler.start_tcp_servers(
                                                &self.unity_launcher.launched_ports, 
                                                &self.unity_launcher.port_algorithm_map,
                                                &self.algorithm_configs,
                                                &self.results_path,
                                                &self.shared_memory_path,
                                                &self.init_path,
                                                self.checkpoint_settings.mode,
                                                &self.device,
                                                self.checkpoint_settings.enable_race_mode,
                                            );
                                            self.show_monitor = true;
                                        }
                                    }
                                    Err(e) => {
                                        self.unity_launcher.is_training_started = false;
                                        rfd::MessageDialog::new().set_title("Error").set_description(&e).show();
                                    }
                                }
                            }

                            if self.unity_launcher.is_training_started {
                                if ui_buttons.button(egui::RichText::new("Stop Training").size(18.0)).clicked() {
                                    self.unity_launcher.stop_training();
                                    self.tcp_server_handler.stop_servers();
                                }
                            }
                        });
                    }
                });
        });
    }

    fn perform_auto_resume_discovery(&mut self) {
        let root = std::path::Path::new(&self.results_path);
        let run_path = if self.run_id.is_empty() {
            root.to_path_buf()
        } else {
            root.join(&self.run_id)
        };

        println!("🔍 Scanning for race checkpoints in: {:?}", run_path);

        if !run_path.exists() {
            rfd::MessageDialog::new().set_title("Error").set_description("Results/Run path not found").show();
            return;
        }

        // Reset counts
        self.ppo_enabled = false; self.sac_enabled = false; self.td3_enabled = false;
        self.tdsac_enabled = false; self.tqc_enabled = false; self.crossq_enabled = false;
        self.drqv2_enabled = false; self.ppo_et_enabled = false; self.ppo_ce_enabled = false;
        
        self.ppo_env_count = 0; self.sac_env_count = 0; self.td3_env_count = 0;
        self.tdsac_env_count = 0; self.tqc_env_count = 0; self.crossq_env_count = 0;
        self.drqv2_env_count = 0; self.ppo_et_env_count = 0; self.ppo_ce_env_count = 0;

        let mut total_found = 0;

        if let Ok(entries) = std::fs::read_dir(&run_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with("race_") {
                            // Check checkpoint subfolder
                            let ckpt_root = path.join("checkpoint");
                            if let Ok(algo_entries) = std::fs::read_dir(ckpt_root) {
                                for algo_entry in algo_entries.flatten() {
                                    if algo_entry.path().is_dir() {
                                        if let Some(algo_name) = algo_entry.file_name().to_str() {
                                            // Check if checkpoint.ot exists
                                            if algo_entry.path().join("checkpoint.ot").exists() {
                                                total_found += 1;
                                                match algo_name.to_lowercase().as_str() {
                                                    "ppo" => { self.ppo_enabled = true; self.ppo_env_count += 1; },
                                                    "sac" => { self.sac_enabled = true; self.sac_env_count += 1; },
                                                    "td3" => { self.td3_enabled = true; self.td3_env_count += 1; },
                                                    "tdsac" => { self.tdsac_enabled = true; self.tdsac_env_count += 1; },
                                                    "tqc" => { self.tqc_enabled = true; self.tqc_env_count += 1; },
                                                    "crossq" => { self.crossq_enabled = true; self.crossq_env_count += 1; },
                                                    "drqv2" => { self.drqv2_enabled = true; self.drqv2_env_count += 1; },
                                                    "ppo_et" => { self.ppo_et_enabled = true; self.ppo_et_env_count += 1; },
                                                    "ppo_ce" => { self.ppo_ce_enabled = true; self.ppo_ce_env_count += 1; },
                                                    _ => println!("⚠️ Unknown algo in checkpoint: {}", algo_name),
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if total_found == 0 {
            rfd::MessageDialog::new().set_title("Result").set_description("No valid race checkpoints found.").show();
            return;
        }

        self.total_env = total_found as u16;
        self.distributed_env_total = total_found;
        self.algorithm_selection_mode = crate::pages::config_types::AlgorithmSelectionMode::Different;
        
        // Transition UI
        self.current_config_section = ConfigSection::Algorithms;
        self.settings_open = false;
        self.algorithms_open = true;
        self.algorithm_config_step = AlgorithmConfigStep::Selection;

        rfd::MessageDialog::new().set_title("Success").set_description(&format!("Found {} environments. Configured.", total_found)).show();
    }
}

impl Drop for HomePage {
    fn drop(&mut self) {
        println!("\n🛑 Closing Orchestrator - Starting cleanup...");
        let total_processes = self.unity_launcher.unity_processes.len();
        if total_processes > 0 {
            for mut child in self.unity_launcher.unity_processes.drain(..) {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        println!("✅ Cleanup completed.\n");
    }
}
