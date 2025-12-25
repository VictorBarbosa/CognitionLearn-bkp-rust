use eframe::egui;
use rfd::FileDialog;
use crate::pages::config_types::{ConfigSection, CheckpointMode, EngineSettings, CheckpointSettings};

// Settings UI logic
pub fn render_settings_ui(
    ui: &mut egui::Ui,
    total_env: &mut u16,
    headless: &mut bool,
    visual_progress: &mut bool,
    device: &mut String,
    run_id: &mut String,
    env_path: &mut String,
    results_path: &mut String,
    base_port: &mut u16,
    learning_rate: &mut f32,
    learning_rate_schedule: &mut String,
    hidden_units: &mut u32,
    num_layers: &mut u32,
    normalize: &mut bool,
    reward_gamma: &mut f32,
    reward_strength: &mut f32,
    max_steps: &mut u64,
    summary_freq: &mut u32,
    init_path: &mut String,
    num_areas: &mut u16,
    timeout_wait: &mut u16,
    seed: &mut i32,
    max_lifetime_restarts: &mut u16,
    restarts_rate_limit_n: &mut u16,
    restarts_rate_limit_period_s: &mut u16,
    engine_settings: &mut EngineSettings,
    checkpoint_settings: &mut CheckpointSettings,
    checkpoint_interval: &mut u32,
    keep_checkpoints: &mut u32,
    current_config_section: &mut ConfigSection,
    settings_open: &mut bool,
    algorithms_open: &mut bool,
) {
    ui.add_space(5.0); // Initial spacing

    egui::Grid::new("general_settings_grid")
        .num_columns(2)
        .spacing([40.0, 4.0])
        .striped(true)
        .show(ui, |ui| {
            ui.label("Total Environments:");
            ui.add(egui::DragValue::new(total_env).speed(1.0).range(1..=u16::MAX));
            ui.end_row();

            ui.label("Headless:");
            ui.checkbox(headless, "");
            ui.end_row();

            ui.label("Visual Progress:");
            ui.checkbox(visual_progress, "");
            ui.end_row();

            ui.label("Device:");
            egui::ComboBox::from_id_salt("device_combo")
                .selected_text(device.to_uppercase())
                .show_ui(ui, |ui| {
                    ui.selectable_value(device, "cpu".to_string(), "CPU");
                    
                    // Logic for macOS (MPS)
                    if cfg!(target_os = "macos") {
                        let mps_available = tch::utils::has_mps();
                        let label = if mps_available { "MPS (Metal)" } else { "MPS (Not Detected)" };
                        if ui.add_enabled(mps_available, egui::SelectableLabel::new(*device == "mps", label)).clicked() {
                            *device = "mps".to_string();
                        }
                    } 
                    
                    // Logic for Linux/Windows (CUDA)
                    if cfg!(target_os = "linux") || cfg!(target_os = "windows") {
                        let cuda_available = tch::Cuda::is_available();
                        let label = if cuda_available { "CUDA (NVIDIA)" } else { "CUDA (Not Detected)" };
                        if ui.add_enabled(cuda_available, egui::SelectableLabel::new(*device == "cuda", label)).clicked() {
                            *device = "cuda".to_string();
                        }
                    }
                });
            ui.end_row();

            ui.label("Run ID:");
            ui.text_edit_singleline(run_id);
            ui.end_row();

            ui.label("Results Path:");
            ui.horizontal(|ui| {
                if ui.button("Browse").clicked() {
                    if let Some(path) = FileDialog::new()
                        .add_filter("Directories", &[""])
                        .pick_folder() {
                        *results_path = path.display().to_string();
                    }
                }
                ui.label(results_path.as_str());
            });
            ui.end_row();

            ui.label("Environment Path:");
            ui.horizontal(|ui| {
                if ui.button("Select").clicked() {
                    if let Some(path) = FileDialog::new()
                        .add_filter("Unity Executables", &["exe", "app", "x86_64", "x86", "out", "bin", "run"])
                        .add_filter("All Executables", &["exe", "app", "x86_64", "x86", "out", "bin", "run", "elf"])
                        .add_filter("All Files", &["*"])
                        .pick_file() {
                        *env_path = path.display().to_string();
                    }
                }
                ui.label(env_path.as_str());
            });
            ui.end_row();

            ui.label("Base Port:");
            ui.add(egui::DragValue::new(base_port).speed(1.0));
            ui.end_row();

            ui.label("Number of Environments:");
            ui.add(egui::DragValue::new(total_env).speed(1.0).range(1..=u16::MAX));
            ui.end_row();

            ui.label("Number of Areas:");
            ui.add(egui::DragValue::new(num_areas).speed(1.0));
            ui.end_row();

            ui.label("Timeout Wait:");
            ui.add(egui::DragValue::new(timeout_wait).speed(1.0));
            ui.end_row();

            ui.label("Seed:");
            ui.add(egui::DragValue::new(seed).speed(1.0));
            ui.end_row();

            ui.label("Max Lifetime Restarts:");
            ui.add(egui::DragValue::new(max_lifetime_restarts).speed(1.0));
            ui.end_row();

            ui.label("Restarts Rate Limit N:");
            ui.add(egui::DragValue::new(restarts_rate_limit_n).speed(1.0));
            ui.end_row();

            ui.label("Restarts Rate Limit Period S:");
            ui.add(egui::DragValue::new(restarts_rate_limit_period_s).speed(1.0));
            ui.end_row();
        });

    ui.add_space(10.0);

    // --- Global Hyperparameters Section ---
    egui::CollapsingHeader::new(egui::RichText::new("Global Hyperparameters").size(18.0))
        .default_open(true)
        .show(ui, |ui| {
            egui::Grid::new("global_hparams_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Learning Rate:");
                    ui.add(egui::DragValue::new(learning_rate).speed(0.0001).range(0.0..=1.0));
                    ui.end_row();

                    ui.label("LR Schedule:");
                    ui.text_edit_singleline(learning_rate_schedule);
                    ui.end_row();

                    ui.label("Hidden Units:");
                    ui.add(egui::DragValue::new(hidden_units).speed(16).range(1..=4096));
                    ui.end_row();

                    ui.label("Num Layers:");
                    ui.add(egui::DragValue::new(num_layers).speed(1).range(1..=32));
                    ui.end_row();

                    ui.label("Normalize Observations:");
                    ui.checkbox(normalize, "");
                    ui.end_row();

                    ui.label("Reward Gamma:");
                    ui.add(egui::DragValue::new(reward_gamma).speed(0.001).range(0.0..=1.0));
                    ui.end_row();

                    ui.label("Reward Strength:");
                    ui.add(egui::DragValue::new(reward_strength).speed(0.1));
                    ui.end_row();

                    ui.label("Max Steps:");
                    ui.add(egui::DragValue::new(max_steps).speed(1000));
                    ui.end_row();

                    ui.label("Summary Frequency:");
                    ui.add(egui::DragValue::new(summary_freq).speed(100));
                    ui.end_row();

                    ui.label("Init Path (Checkpoint):");
                    ui.horizontal(|ui| {
                        if ui.button("Select File").clicked() {
                            if let Some(path) = FileDialog::new().pick_file() {
                                *init_path = path.display().to_string();
                            }
                        }
                        ui.label(init_path.as_str());
                    });
                    ui.end_row();
                });
        });

    ui.add_space(10.0); // Spacing between sections

    // Engine Settings
    egui::CollapsingHeader::new(egui::RichText::new("Engine Settings").size(18.0))
        .show(ui, |ui| {
            egui::Grid::new("engine_settings_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Width:");
                    ui.add(egui::DragValue::new(&mut engine_settings.width).speed(1.0));
                    ui.end_row();

                    ui.label("Height:");
                    ui.add(egui::DragValue::new(&mut engine_settings.height).speed(1.0));
                    ui.end_row();

                    ui.label("Quality Level:");
                    ui.add(egui::DragValue::new(&mut engine_settings.quality_level).speed(1.0));
                    ui.end_row();

                    ui.label("Time Scale:");
                    ui.add(egui::Slider::new(&mut engine_settings.time_scale, 1.0..=100.0));
                    ui.end_row();

                    ui.label("Target Frame Rate:");
                    ui.add(egui::DragValue::new(&mut engine_settings.target_frame_rate).speed(1.0));
                    ui.end_row();

                    ui.label("Capture Frame Rate:");
                    ui.add(egui::DragValue::new(&mut engine_settings.capture_frame_rate).speed(1.0));
                    ui.end_row();

                    ui.label("No Graphics:");
                    ui.checkbox(&mut engine_settings.no_graphics, "");
                    ui.end_row();
                });
        });
    ui.add_space(10.0);

    // Checkpoint Settings
    egui::CollapsingHeader::new(egui::RichText::new("Checkpoint Settings").size(18.0))
        .show(ui, |ui| {
            egui::Grid::new("checkpoint_settings_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Run ID:");
                    ui.text_edit_singleline(&mut checkpoint_settings.run_id);
                    ui.end_row();

                    ui.label("Mode:");
                    ui.vertical(|ui| { // Use a vertical layout for the radio buttons
                        ui.radio_value(&mut checkpoint_settings.mode, CheckpointMode::None, "None");
                        ui.radio_value(&mut checkpoint_settings.mode, CheckpointMode::LoadModel, "Load Model");
                        ui.radio_value(&mut checkpoint_settings.mode, CheckpointMode::Resume, "Resume");
                        ui.radio_value(&mut checkpoint_settings.mode, CheckpointMode::Force, "Force");
                        ui.radio_value(&mut checkpoint_settings.mode, CheckpointMode::TrainModel, "Train Model");
                        ui.radio_value(&mut checkpoint_settings.mode, CheckpointMode::Inference, "Inference");
                    });
                    ui.end_row();

                    ui.label("Results Directory:");
                    ui.text_edit_singleline(&mut checkpoint_settings.results_dir);
                    ui.end_row();

                    ui.label("Default Checkpoint Interval:");
                    ui.add(egui::DragValue::new(checkpoint_interval).speed(100));
                    ui.end_row();

                    ui.label("Default Keep Checkpoints:");
                    ui.add(egui::DragValue::new(keep_checkpoints).speed(1));
                    ui.end_row();

                    ui.label("Race Mode (Tournament):");
                    ui.checkbox(&mut checkpoint_settings.enable_race_mode, "Enable Survival of the Fittest");
                    ui.end_row();
                });
        });
    ui.add_space(10.0);

    ui.add_space(20.0); // Space before the button to separate it from the last fields
    ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui_button| {
        let label = if checkpoint_settings.mode == CheckpointMode::Resume {
            "Proceed to Algorithm Check ➡️"
        } else {
            "Next ➡️"
        };

        if ui_button.button(egui::RichText::new(label).size(18.0)).clicked() {
            *current_config_section = ConfigSection::Algorithms; // Transition to Algorithms
            *settings_open = false; // Close Settings (now correctly false)
            *algorithms_open = true; // Open Algorithms
        }
    });
}