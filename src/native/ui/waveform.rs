use super::theme;
use eframe::egui;

pub fn draw_waveform(ui: &mut egui::Ui, samples: &[f32], sample_rate: u32) {
    let height = (ui.ctx().screen_rect().height() - 500.0).clamp(64.0, 110.0);
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 8, theme::BG);
    let area = rect.shrink(16.0);
    let count = (area.width() / 6.0).max(1.0) as usize;
    let start = samples.len().saturating_sub(sample_rate as usize * 2);
    let visible = &samples[start..];
    for bar in 0..count {
        let from = bar * visible.len() / count;
        let to = (bar + 1) * visible.len() / count;
        let peak = visible[from..to]
            .iter()
            .fold(0.0_f32, |acc, s| acc.max(s.abs()))
            .min(1.0);
        let height = (peak * area.height()).max(2.0);
        let x = area.left() + (bar as f32 + 0.5) * area.width() / count as f32;
        let bar_rect =
            egui::Rect::from_center_size(egui::pos2(x, area.center().y), egui::vec2(3.0, height));
        painter.rect_filled(
            bar_rect,
            2,
            if peak > 0.01 {
                theme::ACCENT
            } else {
                theme::BORDER
            },
        );
    }
}
