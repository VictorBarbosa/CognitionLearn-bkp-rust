use eframe::egui;
use egui::{Context, TextureHandle};
use egui::load::SizedTexture;
use std::time::Instant;

pub struct LoadingPage {
    splash: TextureHandle,
    splash_timer_start_time: Option<Instant>,
    splash_duration_seconds: f32,
}

impl LoadingPage {
    pub fn new(ctx: &Context) -> Self {
        let splash = Self::load_logo(ctx);

        Self {
            splash,
            splash_timer_start_time: None,
            splash_duration_seconds: 5.0,
        }
    }

    fn load_logo(ctx: &Context) -> TextureHandle {
        // Embed file in binary
        const BYTES: &[u8] = include_bytes!("../assets/logo.png"); // Corrected path

        let mut image = image::load_from_memory(BYTES)
            .expect("Invalid logo.png")
            .to_rgba8();

        let mut width = image.width();
        let mut height = image.height();

        let max_texture_side = 2048; // Max size for egui textures

        if width > max_texture_side || height > max_texture_side {
            let scale_factor = (max_texture_side as f32 / width as f32)
                .min(max_texture_side as f32 / height as f32);

            width = (width as f32 * scale_factor).round() as u32;
            height = (height as f32 * scale_factor).round() as u32;

            image = image::imageops::resize(&image, width, height, image::imageops::FilterType::Lanczos3);
        }

        let size = [width as usize, height as usize]; // Usar as novas dimensões

        let img = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());

        ctx.load_texture("logo", img, Default::default())
    }

    pub fn update(&mut self, ctx: &egui::Context, show_splash_state: &mut bool) {
        let mut visuals = egui::Visuals::default();
        visuals.window_fill = egui::Color32::BLACK;
        visuals.panel_fill = egui::Color32::BLACK;
        visuals.widgets.noninteractive.fg_stroke.color = egui::Color32::YELLOW; // Text color for labels
        ctx.set_visuals(visuals);

        // Start the timer if it hasn't started yet
        if self.splash_timer_start_time.is_none() {
            self.splash_timer_start_time = Some(Instant::now());
        }

        let elapsed_time = self.splash_timer_start_time.unwrap().elapsed().as_secs_f32();
        let remaining_time = self.splash_duration_seconds - elapsed_time;

        if remaining_time <= 0.0 {
            *show_splash_state = false;
            ctx.set_visuals(egui::Visuals::dark()); // Restore to dark mode
            return;
        }

        egui::CentralPanel::default().show(ctx, |ui_main| {
            ui_main.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                ui.add_space(200.0); // Space to push the block down and center the image vertically

                let tex = &self.splash;
                let size = tex.size();
                let original_w = size[0] as f32;
                let original_h = size[1] as f32;
                let max_w = 900.0;
                let scale = max_w / original_w;
                let new_size = egui::Vec2::new(original_w * scale, original_h * scale);
                ui.image(SizedTexture::new(tex.id(), new_size));

                ui.add_space(30.0); // Espaço entre a imagem e o contador

                ui.label(egui::RichText::new(format!("Starting in {:.1}...", remaining_time)).color(egui::Color32::YELLOW).size(42.0));
            });
        });
        ctx.request_repaint();
    }
}