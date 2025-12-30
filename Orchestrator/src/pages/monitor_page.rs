use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Legend, Corner, VLine, Text};
use std::sync::mpsc::Receiver;
use crate::trainer::GuiUpdate;
use std::collections::{VecDeque, HashMap, BTreeMap};

struct RaceMilestoneRow {
    index: usize,
    target_step: usize,
    percentage: f32,
    eliminated: String,
    winner: String,
    completed: bool,
}

pub struct MonitorPage {
    // Data stores for plotting
    // Category -> Metric Name -> Behavior Name -> Points [step, value]
    metrics_history: BTreeMap<String, BTreeMap<String, HashMap<String, Vec<[f64; 2]>>>>,
    logs: VecDeque<String>,
    
    // Limits
    max_log_lines: usize,
    
    // UI State
    selected_tab: String,
    
    // Race Info
    race_config: Option<(usize, Vec<f32>)>,
    race_milestones: Vec<RaceMilestoneRow>,
}

impl MonitorPage {
    pub fn new() -> Self {
        Self {
            metrics_history: BTreeMap::new(),
            logs: VecDeque::new(),
            max_log_lines: 1000,
            selected_tab: "Overview".to_string(),
            race_config: None,
            race_milestones: Vec::new(),
        }
    }

    pub fn set_race_config(&mut self, total_steps: usize, num_participants: usize) {
        if num_participants < 2 { return; }
        
        let mut checkpoints = Vec::new();
        for i in 1..num_participants {
            checkpoints.push(i as f32 / num_participants as f32);
        }
        
        self.race_config = Some((total_steps, checkpoints.clone()));
        self.race_milestones.clear();
        
        for (i, &pct) in checkpoints.iter().enumerate() {
            self.race_milestones.push(RaceMilestoneRow {
                index: i,
                target_step: (pct as f64 * total_steps as f64) as usize,
                percentage: pct,
                eliminated: "Pending".to_string(),
                winner: "-".to_string(),
                completed: false,
            });
        }
        self.add_log(format!("🏁 Race Schedule Pre-calculated: {} participants, {} steps.", num_participants, total_steps));
    }

    pub fn update(&mut self, ui: &mut egui::Ui, receiver: &Receiver<GuiUpdate>) {
        // Poll all available updates non-blocking
        let mut received_any = false;
        while let Ok(msg) = receiver.try_recv() {
            received_any = true;
            match msg {
                GuiUpdate::Log(msg) => {
                    self.add_log(msg);
                }
                GuiUpdate::StepInfo { step, avg_reward, buffer_size, behavior_name } => {
                    let step_f64 = step as f64;
                    
                    // Add to "Overview/Average Reward"
                    self.metrics_history
                        .entry("Overview".to_string())
                        .or_default()
                        .entry("Average Reward".to_string())
                        .or_default()
                        .entry(behavior_name.clone())
                        .or_default()
                        .push([step_f64, avg_reward as f64]);
                        
                    // Add to "Overview/Replay Buffer Size"
                    self.metrics_history
                        .entry("Overview".to_string())
                        .or_default()
                        .entry("Replay Buffer Size".to_string())
                        .or_default()
                        .entry(behavior_name.clone())
                        .or_default()
                        .push([step_f64, buffer_size as f64]);
                    
                    self.add_log(format!("[{}] Step: {}, Reward: {:.4}, Buffer: {}", 
                        behavior_name, step, avg_reward, buffer_size));
                }
                GuiUpdate::TrainingUpdate { step, metrics, behavior_name } => {
                     let step_f64 = step as f64;
                     for (metric_name, value) in metrics {
                         // Parse category from "Category/Name"
                         let parts: Vec<&str> = metric_name.split('/').collect();
                         let (category, display_name) = if parts.len() > 1 {
                             (parts[0].to_string(), parts[1..].join("/"))
                         } else {
                             ("Misc".to_string(), metric_name.clone())
                         };
                         
                         self.metrics_history
                             .entry(category)
                             .or_default()
                             .entry(display_name)
                             .or_default()
                             .entry(behavior_name.clone())
                             .or_default()
                             .push([step_f64, value as f64]);
                     }
                }
                GuiUpdate::RaceConfig { total_steps, checkpoints } => {
                    self.race_config = Some((total_steps, checkpoints.clone()));
                    
                    // Initialize milestones table
                    self.race_milestones.clear();
                    for (i, &pct) in checkpoints.iter().enumerate() {
                        self.race_milestones.push(RaceMilestoneRow {
                            index: i,
                            target_step: (pct as f64 * total_steps as f64) as usize,
                            percentage: pct,
                            eliminated: "Pending".to_string(),
                            winner: "-".to_string(),
                            completed: false,
                        });
                    }
                    
                    self.add_log(format!("🏁 Race Configured: {} steps, Checkpoints: {:?}", total_steps, checkpoints));
                }
                GuiUpdate::RaceRound { milestone_idx, start_step: _, end_step: _, eliminated_id, winner_id } => {
                    if let Some(row) = self.race_milestones.get_mut(milestone_idx) {
                        row.eliminated = eliminated_id.clone();
                        row.winner = winner_id.clone();
                        row.completed = true;
                    }
                    self.add_log(format!("🏁 Round {} Finished. Eliminated: {}, Winner: {}", milestone_idx, eliminated_id, winner_id));
                }
            }
        }

        if received_any {
            ui.ctx().request_repaint();
        }

        // Render UI
        // Logs Panel
        egui::SidePanel::right("monitor_logs_panel")
            .resizable(true)
            .default_width(300.0)
            .show_inside(ui, |ui| {
                ui.heading("Logs");
                ui.separator();
                egui::ScrollArea::vertical()
                    .id_salt("monitor_logs_scroll") // Explicit ID
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                    for log in &self.logs {
                        ui.label(egui::RichText::new(log).size(12.0).monospace());
                    }
                });
            });

        // Main Content
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.horizontal(|ui_head| {
                ui_head.heading("Training Dashboard");
                ui_head.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui_btn| {
                    if ui_btn.button(egui::RichText::new("🗑 Clear Charts").color(egui::Color32::LIGHT_RED)).clicked() {
                        self.metrics_history.clear();
                        self.add_log("🧹 Charts cleared by user.".to_string());
                    }
                });
            });
            ui.separator();
            
            // Collect all available categories + "Overview"
            let mut categories: Vec<String> = self.metrics_history.keys().cloned().collect();
            // Ensure Overview is first
            if !categories.contains(&"Overview".to_string()) {
                categories.insert(0, "Overview".to_string());
            } else {
                // Move Overview to front if it exists but isn't first (BTreeMap sorts alphabetically)
                // "Overview" starts with 'O', usually after "Losses".
                categories.retain(|c| c != "Overview");
                categories.insert(0, "Overview".to_string());
            }
            // Add Race tab explicitly
            categories.insert(1, "Race".to_string());

            // Tabs
            egui::ScrollArea::horizontal()
                .id_salt("monitor_tabs_scroll") // Explicit ID
                .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for cat in &categories {
                        if ui.selectable_label(self.selected_tab == *cat, cat).clicked() {
                            self.selected_tab = cat.clone();
                        }
                    }
                });
            });
            ui.separator();

            // Content for selected tab
            egui::ScrollArea::vertical()
                .id_salt("monitor_content_scroll") // Explicit ID
                .show(ui, |ui| {
                
                if self.selected_tab == "Race" {
                     // Custom Race View
                     
                     // 1. Progress Bar (if race config available)
                     if let Some((total_steps, _)) = &self.race_config {
                         // Calculate max current step from data
                         let mut max_current_step = 0.0;
                         if let Some(overview_map) = self.metrics_history.get("Overview") {
                             if let Some(reward_data) = overview_map.get("Average Reward") {
                                 for points in reward_data.values() {
                                     if let Some(last) = points.last() {
                                         if last[0] > max_current_step { max_current_step = last[0]; }
                                     }
                                 }
                             }
                         }
                         
                         let progress = (max_current_step / *total_steps as f64).clamp(0.0, 1.0);
                         ui.add(egui::ProgressBar::new(progress as f32)
                            .text(format!("Race Progress: {:.1}% (Step {:.0}/{})", progress * 100.0, max_current_step, total_steps))
                            .animate(true)
                         );
                         ui.add_space(20.0);
                     }

                     // 2. Race Grid (Table)
                     ui.label(egui::RichText::new("🏆 Race Schedule & Results").strong().size(20.0));
                     ui.add_space(10.0);
                     
                     egui::Grid::new("race_milestones_grid")
                        .striped(true)
                        .spacing([40.0, 10.0])
                        .min_col_width(100.0)
                        .show(ui, |ui| {
                            // Headers
                            ui.label(egui::RichText::new("Phase").strong().size(14.0));
                            ui.label(egui::RichText::new("Target Step").strong().size(14.0));
                            ui.label(egui::RichText::new("Eliminated").strong().color(egui::Color32::LIGHT_RED).size(14.0));
                            ui.label(egui::RichText::new("Best Current").strong().color(egui::Color32::GREEN).size(14.0));
                            ui.end_row();
                            
                            // Data
                            for item in &self.race_milestones {
                                ui.label(format!("Phase {} ({:.0}%)", item.index + 1, item.percentage * 100.0));
                                ui.label(format!("{}", item.target_step));
                                
                                if item.completed {
                                    ui.label(egui::RichText::new(&item.eliminated).color(egui::Color32::LIGHT_RED));
                                    ui.label(egui::RichText::new(&item.winner).color(egui::Color32::GREEN));
                                } else {
                                    ui.label(egui::RichText::new("Pending").italics().color(egui::Color32::GRAY));
                                    ui.label(egui::RichText::new("-").color(egui::Color32::GRAY));
                                }
                                ui.end_row();
                            }
                        });
                     
                     if self.race_milestones.is_empty() {
                         ui.label("Waiting for configuration...");
                     }

                } else if let Some(metrics_map) = self.metrics_history.get(&self.selected_tab) {
                    for (metric_name, behaviors) in metrics_map {
                        ui.label(egui::RichText::new(metric_name).strong().size(16.0));
                        
                        Plot::new(format!("plot_{}_{}", self.selected_tab, metric_name))
                            .height(420.0) // Increased height by 40% (300 * 1.4)
                            .legend(Legend::default().position(Corner::LeftTop)) // LeftTop Legend
                            .show(ui, |plot_ui| {
                                for (behavior, points) in behaviors {
                                    plot_ui.line(Line::new(PlotPoints::new(points.clone())).name(behavior));
                                }
                            });
                        
                        ui.add_space(20.0);
                    }
                } else {
                    ui.label("No data for this category yet.");
                }
            });
        });
    }

    fn add_log(&mut self, msg: String) {
        if self.logs.len() >= self.max_log_lines {
            self.logs.pop_front();
        }
        self.logs.push_back(msg);
    }
}