use eframe::{egui, App};
use egui::Context;
use crate::pages::{loading_page::LoadingPage, home_page::HomePage};

pub struct MyApp {
    show_splash: bool, // Agora só controla qual página exibir
    loading_page: LoadingPage,
    home_page: HomePage,
}

impl MyApp {
    pub fn new(ctx: &Context) -> Self {
        Self {
            show_splash: true, // Começa mostrando a splash screen
            loading_page: LoadingPage::new(ctx), // Inicializa a página de carregamento
            home_page: HomePage::new(), // Inicializa a página principal
        }
    }
}

impl App for MyApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        if self.show_splash {
            self.loading_page.update(ctx, &mut self.show_splash);
        } else {
            self.home_page.update(ctx);
        }
    }
}
