use super::theme;
use eframe::egui;

pub fn draw_waveform(ui: &mut egui::Ui, samples: &[f32], sample_rate: u32, height: f32) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.hline(
        rect.x_range(),
        rect.center().y,
        egui::Stroke::new(1.0_f32, theme::BORDER),
    );
    let count = (rect.width() / 5.0).max(1.0) as usize;
    let start = samples.len().saturating_sub(sample_rate as usize * 2);
    let visible = &samples[start..];
    for bar in 0..count {
        let from = bar * visible.len() / count;
        let to = (bar + 1) * visible.len() / count;
        let peak = visible[from..to]
            .iter()
            .fold(0.0_f32, |acc, s| acc.max(s.abs()))
            .min(1.0);
        if peak <= 0.01 {
            continue;
        }
        let x = rect.left() + (bar as f32 + 0.5) * rect.width() / count as f32;
        let bar_rect = egui::Rect::from_center_size(
            egui::pos2(x, rect.center().y),
            egui::vec2(2.0, (peak * rect.height()).max(2.0)),
        );
        painter.rect_filled(bar_rect, 1, theme::ACCENT);
    }
}
