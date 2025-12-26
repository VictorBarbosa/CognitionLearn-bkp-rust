use eframe::egui;
use std::collections::HashMap;
use crate::pages::config_types::{AlgorithmConfigStep, AlgorithmSelectionMode, AlgoConfig, ConfigSection};

// Algorithm UI logic
pub fn render_algorithm_selection_ui(
    ui: &mut egui::Ui,
    total_env: u16,
    algorithm_selection_mode: &mut AlgorithmSelectionMode,
    ppo_enabled: &mut bool,
    sac_enabled: &mut bool,
    td3_enabled: &mut bool,
    tdsac_enabled: &mut bool,
    tqc_enabled: &mut bool,
    crossq_enabled: &mut bool,
    drqv2_enabled: &mut bool,
    ppo_et_enabled: &mut bool,
    ppo_ce_enabled: &mut bool,
    ppo_env_count: &mut u32,
    sac_env_count: &mut u32,
    td3_env_count: &mut u32,
    tdsac_env_count: &mut u32,
    tqc_env_count: &mut u32,
    crossq_env_count: &mut u32,
    drqv2_env_count: &mut u32,
    ppo_et_env_count: &mut u32,
    ppo_ce_env_count: &mut u32,
    distributed_env_total: &mut u32,
    algorithm_config_step: &mut AlgorithmConfigStep,
    current_config_section: &mut ConfigSection,
    settings_open: &mut bool,
    algorithms_open: &mut bool,
    selected_algorithms_for_config: &mut Vec<String>,
    current_config_algorithm_index: &mut usize,
    algorithm_configs: &mut HashMap<String, AlgoConfig>,
) {
    if *algorithm_config_step == AlgorithmConfigStep::Selection {
        // STEP 1: Algorithm Selection (existing code with modifications)
        if total_env > 1 {
            ui.label(egui::RichText::new("Algorithm Selection Mode").size(16.0));
            ui.add_space(5.0);
            ui.horizontal(|ui_horizontal| {
                ui_horizontal.radio_value(algorithm_selection_mode, AlgorithmSelectionMode::Same, "Same algorithm");
                ui_horizontal.radio_value(algorithm_selection_mode, AlgorithmSelectionMode::Different, "Different algorithms");
            });
        } else {
            *algorithm_selection_mode = AlgorithmSelectionMode::Same;
        }
        ui.separator();
        ui.add_space(10.0);

        if *algorithm_selection_mode != AlgorithmSelectionMode::None {
            // ... (validation logic and selection display - with modified Next button)
            // (The code in this section is long, so I'll paste the complete modified version)
            ui.label(egui::RichText::new(format!("Total Environments (from Settings): {}", total_env)).size(16.0));
            ui.separator();
            ui.add_space(10.0);
            if *algorithm_selection_mode == AlgorithmSelectionMode::Different && total_env > 1 {
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new(format!(
                        "You have {} environments to distribute. Remaining: {}",
                        total_env, (total_env as u32).saturating_sub(*distributed_env_total)
                    )).size(14.0).strong());
                    ui.add_space(10.0);
                });
            }

            let selected_algorithms_count = [
                *ppo_enabled, *sac_enabled, *td3_enabled, *tdsac_enabled,
                *tqc_enabled, *crossq_enabled, *drqv2_enabled,
                *ppo_et_enabled, *ppo_ce_enabled,
            ].iter().filter(|&&x| x).count();

            let can_proceed = if *algorithm_selection_mode == AlgorithmSelectionMode::Same {
                selected_algorithms_count == 1
            } else if *algorithm_selection_mode == AlgorithmSelectionMode::Different && total_env > 1 {
                *distributed_env_total == total_env as u32 && selected_algorithms_count > 0
            } else { // Single env mode
                selected_algorithms_count == 1
            };

            if !can_proceed {
                // ... (error/warning messages)
            }

            // Grid logic for selection (identical to previous)
            egui::Grid::new("algorithms_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    if *algorithm_selection_mode == AlgorithmSelectionMode::Different && total_env > 1 {
                        *distributed_env_total = 0;
                        egui::Grid::new("algorithms_distribution_grid")
                            .num_columns(3)
                            .spacing([40.0, 4.0])
                            .striped(true)
                            .show(ui, |ui_dist| {
                                let mut render_algorithm_row = |ui: &mut egui::Ui, name: &str, enabled: &mut bool, count: &mut u32| {
                                    if ui.checkbox(enabled, "").changed() && !*enabled { *count = 0; }
                                    ui.label(name);
                                    if *enabled {
                                        ui.add(egui::DragValue::new(count).speed(1.0).range(0..=total_env as u32));
                                        *distributed_env_total += *count;
                                    } else {
                                        ui.label("");
                                    }
                                    ui.end_row();
                                };
                                render_algorithm_row(ui_dist, "PPO:", ppo_enabled, ppo_env_count);
                                render_algorithm_row(ui_dist, "SAC:", sac_enabled, sac_env_count);
                                render_algorithm_row(ui_dist, "TD3:", td3_enabled, td3_env_count);
                                render_algorithm_row(ui_dist, "TDSAc:", tdsac_enabled, tdsac_env_count);
                                render_algorithm_row(ui_dist, "TQC:", tqc_enabled, tqc_env_count);
                                render_algorithm_row(ui_dist, "CrossQ:", crossq_enabled, crossq_env_count);
                                render_algorithm_row(ui_dist, "DrQV2:", drqv2_enabled, drqv2_env_count);
                                render_algorithm_row(ui_dist, "PPO_ET:", ppo_et_enabled, ppo_et_env_count);
                                render_algorithm_row(ui_dist, "PPO_CE:", ppo_ce_enabled, ppo_ce_env_count);
                            });
                    } else { // Same or Single Env
                        // ... (identical radio button logic)
                        let mut selected_alg = "";
                        if *ppo_enabled { selected_alg = "PPO"; } else if *sac_enabled { selected_alg = "SAC"; } else if *td3_enabled { selected_alg = "TD3"; } else if *tdsac_enabled { selected_alg = "TDSAc"; } else if *tqc_enabled { selected_alg = "TQC"; } else if *crossq_enabled { selected_alg = "CrossQ"; } else if *drqv2_enabled { selected_alg = "DrQV2"; } else if *ppo_et_enabled { selected_alg = "PPO_ET"; } else if *ppo_ce_enabled { selected_alg = "PPO_CE"; }
                        let mut alg_changed = |name: &str, ui: &mut egui::Ui| { if ui.radio(selected_alg == name, "").clicked() { *ppo_enabled = name == "PPO"; *sac_enabled = name == "SAC"; *td3_enabled = name == "TD3"; *tdsac_enabled = name == "TDSAc"; *tqc_enabled = name == "TQC"; *crossq_enabled = name == "CrossQ"; *drqv2_enabled = name == "DrQV2"; *ppo_et_enabled = name == "PPO_ET"; *ppo_ce_enabled = name == "PPO_CE"; } };
                        ui.label("PPO:"); alg_changed("PPO", ui); ui.end_row(); ui.label("SAC:"); alg_changed("SAC", ui); ui.end_row(); ui.label("TD3:"); alg_changed("TD3", ui); ui.end_row(); ui.label("TDSAc:"); alg_changed("TDSAc", ui); ui.end_row(); ui.label("TQC:"); alg_changed("TQC", ui); ui.end_row(); ui.label("CrossQ:"); alg_changed("CrossQ", ui); ui.end_row(); ui.label("DrQV2:"); alg_changed("DrQV2", ui); ui.end_row(); ui.label("PPO_ET:"); alg_changed("PPO_ET", ui); ui.end_row(); ui.label("PPO_CE:"); alg_changed("PPO_CE", ui); ui.end_row();
                    }
                });

            ui.add_space(20.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui_buttons| {
                if ui_buttons.add_enabled(can_proceed, egui::Button::new(egui::RichText::new("Next").size(18.0))).clicked() {
                    // **MODIFIED LOGIC HERE**
                    selected_algorithms_for_config.clear();
                    if *ppo_enabled { selected_algorithms_for_config.push("PPO".to_string()); }
                    if *sac_enabled { selected_algorithms_for_config.push("SAC".to_string()); }
                    if *td3_enabled { selected_algorithms_for_config.push("TD3".to_string()); }
                    if *tdsac_enabled { selected_algorithms_for_config.push("TDSAc".to_string()); }
                    if *tqc_enabled { selected_algorithms_for_config.push("TQC".to_string()); }
                    if *crossq_enabled { selected_algorithms_for_config.push("CrossQ".to_string()); }
                    if *drqv2_enabled { selected_algorithms_for_config.push("DrQV2".to_string()); }
                    if *ppo_et_enabled { selected_algorithms_for_config.push("PPO_ET".to_string()); }
                    if *ppo_ce_enabled { selected_algorithms_for_config.push("PPO_CE".to_string()); }

                    *current_config_algorithm_index = 0;
                    *algorithm_config_step = AlgorithmConfigStep::Configuration;
                }
                if ui_buttons.button(egui::RichText::new("Back").size(18.0)).clicked() {
                    *current_config_section = crate::pages::config_types::ConfigSection::Settings;
                    *settings_open = true;
                    *algorithms_open = false;
                }
            });
        }
    } else {
        // STEP 2: Sequential Configuration
        if selected_algorithms_for_config.is_empty() {
            ui.label("No algorithms were selected.");
            if ui.button("Back to Selection").clicked() {
                *algorithm_config_step = AlgorithmConfigStep::Selection;
            }
            return;
        }

        let current_algo_name = &selected_algorithms_for_config[*current_config_algorithm_index];
        ui.label(egui::RichText::new(format!("Configuring: {} ({}/{})", current_algo_name, *current_config_algorithm_index + 1, selected_algorithms_for_config.len())).size(16.0));
        ui.separator();
        ui.add_space(10.0);

        // Render the accordion for the current algorithm
        let (count, name) = match current_algo_name.as_str() {
            "PPO" => (*ppo_env_count, "PPO"),
            "SAC" => (*sac_env_count, "SAC"),
            "TD3" => (*td3_env_count, "TD3"),
            "TDSAc" => (*tdsac_env_count, "TDSAc"),
            "TQC" => (*tqc_env_count, "TQC"),
            "CrossQ" => (*crossq_env_count, "CrossQ"),
            "DrQV2" => (*drqv2_env_count, "DrQV2"),
            "PPO_ET" => (*ppo_et_env_count, "PPO_ET"),
            "PPO_CE" => (*ppo_ce_env_count, "PPO_CE"),
            _ => (0, "Unknown"),
        };

        let header_text = if *algorithm_selection_mode == AlgorithmSelectionMode::Different {
            format!("{} ({})", name, count)
        } else {
            name.to_string()
        };

        egui::CollapsingHeader::new(egui::RichText::new(header_text).size(18.0))
            .default_open(true)
            .show(ui, |ui| {
                if let Some(cfg) = algorithm_configs.get_mut(name) {
                    egui::Grid::new(format!("{}_config", name)).num_columns(2).spacing([40.0,4.0]).striped(true).show(ui, |ui_cfg| {
                        // Dynamic Labels
                        let is_on_policy = matches!(name, "PPO" | "PPO_ET" | "PPO_CE" | "POCA");
                        let buffer_label = if is_on_policy { "Horizon (Steps):" } else { "Replay Buffer Capacity:" };
                        let batch_label = if is_on_policy { "Minibatch Size:" } else { "Batch Size:" };
                        let ent_coef_label = if is_on_policy { "Entropy Coef:" } else { "Alpha / Ent Coef:" };

                        ui_cfg.label(batch_label).on_hover_text("Number of experiences used in one gradient update. Larger batches provide more stable gradients but require more memory."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.batch_size).speed(1)); ui_cfg.end_row();

                        ui_cfg.label(buffer_label).on_hover_text(if is_on_policy { "Number of steps collected before updating the policy." } else { "Total number of experiences stored in the replay buffer." }); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.buffer_size).speed(1)); ui_cfg.end_row();

                        ui_cfg.label("Learning Rate:").on_hover_text("Step size for optimizer updates. Too high might cause instability; too low might slow down training."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.learning_rate).speed(0.0001).range(0.0..=1.0)); ui_cfg.end_row();

                        ui_cfg.label("LR Schedule:").on_hover_text("How the learning rate changes over time. 'linear' decays it to zero at max steps."); 
                        ui_cfg.text_edit_singleline(&mut cfg.learning_rate_schedule); ui_cfg.end_row();

                        ui_cfg.label("Hidden Units:").on_hover_text("Number of neurons per hidden layer in the neural network."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.hidden_units).speed(16).range(1..=4096)); ui_cfg.end_row();

                        ui_cfg.label("Num Layers:").on_hover_text("Number of hidden layers in the network architecture."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.num_layers).speed(1).range(1..=32)); ui_cfg.end_row();

                        ui_cfg.label("Normalize Obs:").on_hover_text("Automatically scales input observations to a mean of 0 and standard deviation of 1. Recommended for most environments."); 
                        ui_cfg.checkbox(&mut cfg.normalize, ""); ui_cfg.end_row();

                        ui_cfg.label("Gamma:").on_hover_text("Discount factor for future rewards. 0.99 means the agent cares about long-term rewards; 0.1 means only immediate ones."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.gamma).speed(0.001).range(0.0..=1.0)); ui_cfg.end_row();

                        ui_cfg.label("Strength:").on_hover_text("Multiplier for the extrinsic reward signal."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.strength).speed(0.1)); ui_cfg.end_row();

                        ui_cfg.label("Max Steps:").on_hover_text("Total number of environment steps to run before terminating the training process."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.max_steps).speed(1000)); ui_cfg.end_row();

                        ui_cfg.label("Summary Freq:").on_hover_text("Frequency (in steps) to send metrics to Tensorboard logs."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.summary_freq).speed(100)); ui_cfg.end_row();

                        ui_cfg.label("Checkpoint Interval:").on_hover_text("Frequency (in steps) to save model weights to disk."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.checkpoint_interval).speed(100)); ui_cfg.end_row();

                        ui_cfg.label("Keep Checkpoints:").on_hover_text("Maximum number of recent model files to keep on disk."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.keep_checkpoints).speed(1)); ui_cfg.end_row();

                        ui_cfg.label("Buffer Init Steps:").on_hover_text("Steps collected with random actions before training starts to populate the buffer (Off-policy only)."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.buffer_init_steps).speed(100)); ui_cfg.end_row();

                        ui_cfg.label("Tau:").on_hover_text("Soft update coefficient for target networks. Usually 0.005. Controls how fast target networks follow the main ones."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.tau).speed(0.001)); ui_cfg.end_row();

                        ui_cfg.label("Steps/Update:").on_hover_text("Ratio between environment steps and gradient updates. 1.0 means one update per step."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.steps_per_update).speed(0.1)); ui_cfg.end_row();

                        ui_cfg.label("Save Replay:").on_hover_text("Whether to save the Replay Buffer state to disk when closing."); 
                        ui_cfg.checkbox(&mut cfg.save_replay_buffer, ""); ui_cfg.end_row();

                        if let Some(v)=cfg.init_entcoef.as_mut(){ 
                            ui_cfg.label(ent_coef_label).on_hover_text("Initial weight of the entropy bonus. Higher values encourage more exploration."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(0.01)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.lambd.as_mut(){ 
                            ui_cfg.label("GAE Lambda:").on_hover_text("Smoothing factor for Generalized Advantage Estimation. Lower values reduce variance but increase bias."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(0.01)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.num_epoch.as_mut(){ 
                            ui_cfg.label("Num Epochs:").on_hover_text("Number of times to iterate over the data during a single update (PPO)."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.beta.as_mut(){ 
                            ui_cfg.label("Beta:").on_hover_text("Strength of entropy regularization. Helps prevent premature convergence to local optima."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(0.001)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.epsilon.as_mut(){ 
                            ui_cfg.label("Epsilon:").on_hover_text("Clipping range for policy updates. Prevents the policy from changing too much in one step."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(0.01)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.max_grad_norm.as_mut(){ 
                            ui_cfg.label("Max Grad Norm:").on_hover_text("Limits the magnitude of gradient updates to prevent 'exploding gradients' and training collapse."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(0.1)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.policy_delay.as_mut(){ 
                            ui_cfg.label("Policy Delay:").on_hover_text("How many critic updates occur before a single actor update (TD3). Helps stabilize training."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.n_quantiles.as_mut(){ 
                            ui_cfg.label("N Quantiles:").on_hover_text("Number of atoms used for Distributional RL. Higher means more precise value estimation."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.n_to_drop.as_mut(){ 
                            ui_cfg.label("N To Drop:").on_hover_text("Number of top quantiles to discard to prevent overestimation of values (TQC)."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); 
                        }
                        
                        // Remaining optional/specific fields
                        if let Some(v)=cfg.destructive_threshold.as_mut(){ 
                            ui_cfg.label("Destructive Threshold:").on_hover_text("Level of model divergence allowed before discarding a run."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(0.01)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.image_pad.as_mut(){ 
                            ui_cfg.label("Image Pad:").on_hover_text("Pixel padding for visual observations. Acts as data augmentation."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.entropy_temperature.as_mut(){ 
                            ui_cfg.label("Entropy Temp:").on_hover_text("Initial temperature for SAC entropy adjustment."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(0.01)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.adaptive_entropy_temperature.as_mut(){ 
                            ui_cfg.label("Adaptive Entropy:").on_hover_text("Automatically adjust entropy weight to meet a target entropy value."); 
                            ui_cfg.checkbox(v, ""); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.curiosity_strength.as_mut(){ 
                            ui_cfg.label("Curiosity Strength:").on_hover_text("Weight of the intrinsic reward for exploring novel states."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(0.001)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.curiosity_gamma.as_mut(){ 
                            ui_cfg.label("Curiosity Gamma:").on_hover_text("Discount factor for curiosity rewards."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(0.001)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.curiosity_learning_rate.as_mut(){ 
                            ui_cfg.label("Curiosity LR:").on_hover_text("Learning rate for the Intrinsic Curiosity Module (ICM)."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(0.0001)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.curiosity_hidden_units.as_mut(){ 
                            ui_cfg.label("Curiosity Hidden:").on_hover_text("Hidden units in the ICM neural networks."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.curiosity_num_layers.as_mut(){ 
                            ui_cfg.label("Curiosity Layers:").on_hover_text("Number of layers in the ICM neural networks."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.imagination_horizon.as_mut(){ 
                            ui_cfg.label("Imagination Horizon:").on_hover_text("How many steps ahead to simulate internal models for imagination."); 
                            ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); 
                        }
                        if let Some(v)=cfg.use_imagination_augmented.as_mut(){ 
                            ui_cfg.label("Use Imagination Aug:").on_hover_text("Enables augmented training using imagined experiences."); 
                            ui_cfg.checkbox(v, ""); ui_cfg.end_row(); 
                        }
                        
                        // Memory Section
                        ui_cfg.label("Use Memory (LSTM):").on_hover_text("Enables Long Short-Term Memory. Allows the agent to remember past states, essential for partially observable tasks.");
                        if ui_cfg.checkbox(&mut cfg.use_memory, "").changed() {
                            if cfg.use_memory {
                                // Set defaults if newly enabled
                                if cfg.memory_sequence_length.is_none() { cfg.memory_sequence_length = Some(64); }
                                if cfg.memory_size.is_none() { cfg.memory_size = Some(256); }
                            }
                        }
                        ui_cfg.end_row();

                        if cfg.use_memory {
                            if let Some(v) = cfg.memory_sequence_length.as_mut() {
                                ui_cfg.label("Sequence Length:").on_hover_text("Number of past steps the agent considers in its memory.");
                                ui_cfg.add(egui::DragValue::new(v).speed(1));
                                ui_cfg.end_row();
                            }
                            if let Some(v) = cfg.memory_size.as_mut() {
                                ui_cfg.label("Memory Size:").on_hover_text("Number of hidden units in the LSTM memory module.");
                                ui_cfg.add(egui::DragValue::new(v).speed(1));
                                ui_cfg.end_row();
                            }
                        }

                        // Time Horizon
                        ui_cfg.label("Time Horizon:").on_hover_text("Maximum steps per episode before the environment is reset or truncated."); 
                        ui_cfg.add(egui::DragValue::new(&mut cfg.time_horizon).speed(10)); ui_cfg.end_row();
                    });
                } else { ui.label("Config not found"); }
            });

        ui.add_space(20.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui_buttons| {
            // Botão "Next" ou "Start Train"
            if *current_config_algorithm_index < selected_algorithms_for_config.len() - 1 {
                if ui_buttons.button(egui::RichText::new("Next").size(18.0)).clicked() {
                    *current_config_algorithm_index += 1;
                }
            } else {
                // The "Start Train" button logic will be handled in the main HomePage struct
            }

            // Botão "Back" com lógica condicional
            if ui_buttons.button(egui::RichText::new("Back").size(18.0)).clicked() {
                if *current_config_algorithm_index > 0 {
                    *current_config_algorithm_index -= 1;
                } else {
                    *algorithm_config_step = AlgorithmConfigStep::Selection;
                }
            }
        });
    }
}