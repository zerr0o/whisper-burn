use eframe::egui;

const WAVEFORM_COLOR: egui::Color32 = egui::Color32::from_rgb(233, 69, 96); // #e94560

pub fn draw_waveform(ui: &mut egui::Ui, samples: &[f32], sample_rate: u32) {
    let desired_size = egui::vec2(ui.available_width(), 100.0);
    let (rect, _response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

    let painter = ui.painter_at(rect);

    // Draw background
    painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(30, 30, 40));

    if samples.is_empty() {
        return;
    }

    // Show last ~2 seconds
    let display_samples = (sample_rate as usize) * 2;
    let start = if samples.len() > display_samples {
        samples.len() - display_samples
    } else {
        0
    };
    let visible = &samples[start..];

    let width = rect.width() as usize;
    if width == 0 {
        return;
    }

    // Downsample to pixel count
    let samples_per_pixel = visible.len() as f32 / width as f32;

    let mut points = Vec::with_capacity(width);
    for px in 0..width {
        let sample_start = (px as f32 * samples_per_pixel) as usize;
        let sample_end = ((px + 1) as f32 * samples_per_pixel) as usize;
        let sample_end = sample_end.min(visible.len());

        if sample_start >= sample_end {
            continue;
        }

        // Peak value in this bucket
        let peak = visible[sample_start..sample_end]
            .iter()
            .fold(0.0f32, |acc, &s| acc.max(s.abs()));

        let x = rect.left() + px as f32;
        let y = rect.center().y - peak * (rect.height() * 0.45);
        points.push(egui::pos2(x, y));
    }

    // Mirror for bottom half
    let mut all_points = points.clone();
    for p in points.iter().rev() {
        let mirrored_y = rect.center().y + (rect.center().y - p.y);
        all_points.push(egui::pos2(p.x, mirrored_y));
    }

    if all_points.len() >= 3 {
        let stroke = egui::Stroke::new(1.5, WAVEFORM_COLOR);
        painter.add(egui::Shape::line(all_points, stroke));
    }

    // Center line
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.center().y),
            egui::pos2(rect.right(), rect.center().y),
        ],
        egui::Stroke::new(0.5, egui::Color32::from_gray(80)),
    );
}
