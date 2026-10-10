use eframe::egui;

use super::{download_screen, theme};

pub fn draw(ui: &mut egui::Ui, message: &str) {
    download_screen::header(
        ui,
        "Loading model",
        "Preparing the speech model on your GPU.",
        |_| {},
    );

    theme::card().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            theme::spinner(ui, 18.0);
            ui.add(egui::Label::new(egui::RichText::new(message).color(theme::TEXT)).wrap());
        });
    });
}
