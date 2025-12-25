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
    poca_enabled: &mut bool,
    ppo_env_count: &mut u32,
    sac_env_count: &mut u32,
    td3_env_count: &mut u32,
    tdsac_env_count: &mut u32,
    tqc_env_count: &mut u32,
    crossq_env_count: &mut u32,
    drqv2_env_count: &mut u32,
    ppo_et_env_count: &mut u32,
    ppo_ce_env_count: &mut u32,
    poca_env_count: &mut u32,
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
                *ppo_et_enabled, *ppo_ce_enabled, *poca_enabled,
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
                                render_algorithm_row(ui_dist, "POCA:", poca_enabled, poca_env_count);
                            });
                    } else { // Same or Single Env
                        // ... (identical radio button logic)
                        let mut selected_alg = "";
                        if *ppo_enabled { selected_alg = "PPO"; } else if *sac_enabled { selected_alg = "SAC"; } else if *td3_enabled { selected_alg = "TD3"; } else if *tdsac_enabled { selected_alg = "TDSAc"; } else if *tqc_enabled { selected_alg = "TQC"; } else if *crossq_enabled { selected_alg = "CrossQ"; } else if *drqv2_enabled { selected_alg = "DrQV2"; } else if *ppo_et_enabled { selected_alg = "PPO_ET"; } else if *ppo_ce_enabled { selected_alg = "PPO_CE"; } else if *poca_enabled { selected_alg = "POCA"; }
                        let mut alg_changed = |name: &str, ui: &mut egui::Ui| { if ui.radio(selected_alg == name, "").clicked() { *ppo_enabled = name == "PPO"; *sac_enabled = name == "SAC"; *td3_enabled = name == "TD3"; *tdsac_enabled = name == "TDSAc"; *tqc_enabled = name == "TQC"; *crossq_enabled = name == "CrossQ"; *drqv2_enabled = name == "DrQV2"; *ppo_et_enabled = name == "PPO_ET"; *ppo_ce_enabled = name == "PPO_CE"; *poca_enabled = name == "POCA"; } };
                        ui.label("PPO:"); alg_changed("PPO", ui); ui.end_row(); ui.label("SAC:"); alg_changed("SAC", ui); ui.end_row(); ui.label("TD3:"); alg_changed("TD3", ui); ui.end_row(); ui.label("TDSAc:"); alg_changed("TDSAc", ui); ui.end_row(); ui.label("TQC:"); alg_changed("TQC", ui); ui.end_row(); ui.label("CrossQ:"); alg_changed("CrossQ", ui); ui.end_row(); ui.label("DrQV2:"); alg_changed("DrQV2", ui); ui.end_row(); ui.label("PPO_ET:"); alg_changed("PPO_ET", ui); ui.end_row(); ui.label("PPO_CE:"); alg_changed("PPO_CE", ui); ui.end_row(); ui.label("POCA:"); alg_changed("POCA", ui); ui.end_row();
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
                    if *poca_enabled { selected_algorithms_for_config.push("POCA".to_string()); }

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
            "POCA" => (*poca_env_count, "POCA"),
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

                        ui_cfg.label(batch_label); ui_cfg.add(egui::DragValue::new(&mut cfg.batch_size).speed(1)); ui_cfg.end_row();
                        ui_cfg.label(buffer_label); ui_cfg.add(egui::DragValue::new(&mut cfg.buffer_size).speed(1)); ui_cfg.end_row();
                        ui_cfg.label("Learning Rate:"); ui_cfg.add(egui::DragValue::new(&mut cfg.learning_rate).speed(0.0001).range(0.0..=1.0)); ui_cfg.end_row();
                        ui_cfg.label("LR Schedule:"); ui_cfg.text_edit_singleline(&mut cfg.learning_rate_schedule); ui_cfg.end_row();
                        ui_cfg.label("Hidden Units:"); ui_cfg.add(egui::DragValue::new(&mut cfg.hidden_units).speed(16).range(1..=4096)); ui_cfg.end_row();
                        ui_cfg.label("Num Layers:"); ui_cfg.add(egui::DragValue::new(&mut cfg.num_layers).speed(1).range(1..=32)); ui_cfg.end_row();
                        ui_cfg.label("Normalize Obs:"); ui_cfg.checkbox(&mut cfg.normalize, ""); ui_cfg.end_row();
                        ui_cfg.label("Gamma:"); ui_cfg.add(egui::DragValue::new(&mut cfg.gamma).speed(0.001).range(0.0..=1.0)); ui_cfg.end_row();
                        ui_cfg.label("Strength:"); ui_cfg.add(egui::DragValue::new(&mut cfg.strength).speed(0.1)); ui_cfg.end_row();
                        ui_cfg.label("Max Steps:"); ui_cfg.add(egui::DragValue::new(&mut cfg.max_steps).speed(1000)); ui_cfg.end_row();
                        ui_cfg.label("Summary Freq:"); ui_cfg.add(egui::DragValue::new(&mut cfg.summary_freq).speed(100)); ui_cfg.end_row();
                        ui_cfg.label("Checkpoint Interval:"); ui_cfg.add(egui::DragValue::new(&mut cfg.checkpoint_interval).speed(100)); ui_cfg.end_row();
                        ui_cfg.label("Keep Checkpoints:"); ui_cfg.add(egui::DragValue::new(&mut cfg.keep_checkpoints).speed(1)); ui_cfg.end_row();
                        ui_cfg.label("Buffer Init Steps:"); ui_cfg.add(egui::DragValue::new(&mut cfg.buffer_init_steps).speed(100)); ui_cfg.end_row();
                        ui_cfg.label("Tau:"); ui_cfg.add(egui::DragValue::new(&mut cfg.tau).speed(0.001)); ui_cfg.end_row();
                        ui_cfg.label("Steps/Update:"); ui_cfg.add(egui::DragValue::new(&mut cfg.steps_per_update).speed(0.1)); ui_cfg.end_row();
                        ui_cfg.label("Save Replay:"); ui_cfg.checkbox(&mut cfg.save_replay_buffer, ""); ui_cfg.end_row();
                        if let Some(v)=cfg.init_entcoef.as_mut(){ ui_cfg.label(ent_coef_label); ui_cfg.add(egui::DragValue::new(v).speed(0.01)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.lambd.as_mut(){ ui_cfg.label("GAE Lambda:"); ui_cfg.add(egui::DragValue::new(v).speed(0.01)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.num_epoch.as_mut(){ ui_cfg.label("Num Epochs:"); ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.beta.as_mut(){ ui_cfg.label("Beta:"); ui_cfg.add(egui::DragValue::new(v).speed(0.001)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.epsilon.as_mut(){ ui_cfg.label("Epsilon:"); ui_cfg.add(egui::DragValue::new(v).speed(0.01)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.policy_delay.as_mut(){ ui_cfg.label("Policy Delay:"); ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.n_quantiles.as_mut(){ ui_cfg.label("N Quantiles:"); ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.n_to_drop.as_mut(){ ui_cfg.label("N To Drop:"); ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); }
                        
                        // Remaining optional/specific fields
                        if let Some(v)=cfg.destructive_threshold.as_mut(){ ui_cfg.label("Destructive Threshold:"); ui_cfg.add(egui::DragValue::new(v).speed(0.01)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.image_pad.as_mut(){ ui_cfg.label("Image Pad:"); ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.entropy_temperature.as_mut(){ ui_cfg.label("Entropy Temp:"); ui_cfg.add(egui::DragValue::new(v).speed(0.01)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.adaptive_entropy_temperature.as_mut(){ ui_cfg.label("Adaptive Entropy:"); ui_cfg.checkbox(v, ""); ui_cfg.end_row(); }
                        if let Some(v)=cfg.curiosity_strength.as_mut(){ ui_cfg.label("Curiosity Strength:"); ui_cfg.add(egui::DragValue::new(v).speed(0.001)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.curiosity_gamma.as_mut(){ ui_cfg.label("Curiosity Gamma:"); ui_cfg.add(egui::DragValue::new(v).speed(0.001)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.curiosity_learning_rate.as_mut(){ ui_cfg.label("Curiosity LR:"); ui_cfg.add(egui::DragValue::new(v).speed(0.0001)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.curiosity_hidden_units.as_mut(){ ui_cfg.label("Curiosity Hidden:"); ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.curiosity_num_layers.as_mut(){ ui_cfg.label("Curiosity Layers:"); ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.imagination_horizon.as_mut(){ ui_cfg.label("Imagination Horizon:"); ui_cfg.add(egui::DragValue::new(v).speed(1)); ui_cfg.end_row(); }
                        if let Some(v)=cfg.use_imagination_augmented.as_mut(){ ui_cfg.label("Use Imagination Aug:"); ui_cfg.checkbox(v, ""); ui_cfg.end_row(); }
                        
                        // Memory Section
                        ui_cfg.label("Use Memory (LSTM):");
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
                                ui_cfg.label("Sequence Length:");
                                ui_cfg.add(egui::DragValue::new(v).speed(1));
                                ui_cfg.end_row();
                            }
                            if let Some(v) = cfg.memory_size.as_mut() {
                                ui_cfg.label("Memory Size:");
                                ui_cfg.add(egui::DragValue::new(v).speed(1));
                                ui_cfg.end_row();
                            }
                        }

                        // Time Horizon is specific enough (steps per episode limit vs global max steps)
                        ui_cfg.label("Time Horizon:"); ui_cfg.add(egui::DragValue::new(&mut cfg.time_horizon).speed(10)); ui_cfg.end_row();
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