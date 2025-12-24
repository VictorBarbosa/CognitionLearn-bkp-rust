use eframe::egui;
use crate::app::MyApp;

pub fn run() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_maximized(true), // Iniciar maximizado
        ..Default::default()
    };
    eframe::run_native(
        "Orchestrator GUI",
        native_options,
        Box::new(|cc| Ok(Box::new(MyApp::new(&cc.egui_ctx)))),
    )
}
