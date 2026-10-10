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

/// Status dot and label; the dot carries the state colour.
pub fn draw_status(ui: &mut egui::Ui, status: AppStatus) {
    let color = match status {
        AppStatus::Ready | AppStatus::Done => theme::GREEN,
        AppStatus::Recording => theme::RED,
        AppStatus::Processing => theme::ACCENT,
    };
    let galley = ui.painter().layout_no_wrap(
        status.label().to_owned(),
        egui::FontId::proportional(12.5),
        theme::MUTED,
    );
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(galley.size().x + 25.0, 20.0),
        egui::Sense::hover(),
    );
    ui.painter()
        .circle_filled(egui::pos2(rect.left() + 9.5, rect.center().y), 3.0, color);
    ui.painter().galley(
        egui::pos2(rect.left() + 19.0, rect.center().y - galley.size().y / 2.0),
        galley,
        theme::MUTED,
    );
}
