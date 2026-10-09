use eframe::egui;

use super::{download_screen, theme};

pub fn draw(ui: &mut egui::Ui, message: &str) {
    download_screen::header(
        ui,
        "WHISPER BURN",
        "Getting ready",
        "Local GPU inference. Audio stays on this device.",
    );

    theme::card().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            theme::spinner(ui);
            ui.add_space(4.0);
            ui.add(
                egui::Label::new(egui::RichText::new(message).size(16.0).color(theme::TEXT)).wrap(),
            );
        });
    });
}
