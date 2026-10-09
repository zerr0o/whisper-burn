use super::theme;
use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppStatus {
    Ready,
    Recording,
    Processing,
    Done,
}

impl AppStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ready => "Ready",
            Self::Recording => "Recording",
            Self::Processing => "Transcribing",
            Self::Done => "Transcribed",
        }
    }
}

pub fn draw_status(ui: &mut egui::Ui, status: AppStatus) {
    let color = match status {
        AppStatus::Ready => theme::GREEN,
        AppStatus::Recording => theme::ACCENT,
        AppStatus::Processing | AppStatus::Done => theme::ACCENT,
    };
    egui::Frame::new()
        .fill(theme::SURFACE)
        .corner_radius(7)
        .inner_margin(egui::Margin::symmetric(10, 7))
        .show(ui, |ui| {
            ui.spacing_mut().interact_size.y = 20.0;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let (rect, _) = ui.allocate_exact_size(egui::vec2(7.0, 12.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 3.0, color);
                ui.label(egui::RichText::new(status.label()).size(12.0).color(color));
            });
        });
}
