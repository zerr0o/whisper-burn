use eframe::egui;

pub fn draw(ui: &mut egui::Ui, message: &str) {
    ui.vertical_centered(|ui| {
        ui.add_space(120.0);
        ui.spinner();
        ui.add_space(16.0);
        ui.label(egui::RichText::new(message).size(18.0));
    });
}
