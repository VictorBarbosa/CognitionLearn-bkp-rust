use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Legend, Corner, VLine, Text};
use std::sync::mpsc::Receiver;
use crate::trainer::GuiUpdate;
use std::collections::{VecDeque, HashMap, BTreeMap};

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
}

impl MonitorPage {
    pub fn new() -> Self {
        Self {
            metrics_history: BTreeMap::new(),
            logs: VecDeque::new(),
            max_log_lines: 1000,
            selected_tab: "Overview".to_string(),
            race_config: None,
        }
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
                    self.add_log(format!("🏁 Race Configured: {} steps, Checkpoints: {:?}", total_steps, checkpoints));
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
            ui.heading("Training Dashboard");
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
                         ui.add_space(10.0);
                     }

                     // 2. Race Chart with Milestones
                     if let Some(overview_map) = self.metrics_history.get("Overview") {
                         if let Some(reward_data) = overview_map.get("Average Reward") {
                             ui.label(egui::RichText::new("🏆 Race Progress (Average Reward)").strong().size(20.0));
                             
                             Plot::new("plot_race_reward")
                                .height(500.0)
                                .legend(Legend::default().position(Corner::LeftTop))
                                .show(ui, |plot_ui| {
                                    // Plot Agent Lines
                                    for (behavior, points) in reward_data {
                                        plot_ui.line(Line::new(PlotPoints::new(points.clone())).name(behavior).width(2.0));
                                    }
                                    
                                    // Plot Milestones
                                    if let Some((total_steps, checkpoints)) = &self.race_config {
                                        for (i, &pct) in checkpoints.iter().enumerate() {
                                            let step_x = *total_steps as f64 * pct as f64;
                                            plot_ui.vline(VLine::new(step_x).color(egui::Color32::YELLOW).style(egui_plot::LineStyle::Dashed { length: 10.0 }));
                                            
                                            // Add label slightly offset
                                            plot_ui.text(Text::new(
                                                egui_plot::PlotPoint::new(step_x, 0.0), // Y=0 anchor
                                                format!("Phase {} ({:.0}%)", i + 1, pct * 100.0)
                                            ).color(egui::Color32::YELLOW).anchor(egui::Align2::LEFT_BOTTOM));
                                        }
                                        
                                        // Finish Line
                                        plot_ui.vline(VLine::new(*total_steps as f64).color(egui::Color32::GREEN).width(2.0));
                                        plot_ui.text(Text::new(
                                            egui_plot::PlotPoint::new(*total_steps as f64, 0.0),
                                            "FINISH"
                                        ).color(egui::Color32::GREEN).anchor(egui::Align2::RIGHT_BOTTOM));
                                    }
                                });
                         } else {
                             ui.label("Waiting for reward data to start the race visualization...");
                         }
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