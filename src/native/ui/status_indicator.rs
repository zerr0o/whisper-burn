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
            Self::Recording => "Recording...",
            Self::Processing => "Processing...",
            Self::Done => "Done",
        }
    }

    fn color(self) -> egui::Color32 {
        match self {
            Self::Ready => egui::Color32::from_rgb(80, 200, 120),     // green
            Self::Recording => egui::Color32::from_rgb(233, 69, 96),  // red
            Self::Processing => egui::Color32::from_rgb(240, 180, 40), // yellow
            Self::Done => egui::Color32::from_rgb(80, 140, 230),      // blue
        }
    }
}

pub fn draw_status(ui: &mut egui::Ui, status: AppStatus) {
    let color = status.color();

    // Pulsing for recording
    let alpha = if status == AppStatus::Recording {
        let t = ui.input(|i| i.time);
        let pulse = ((t * 3.0).sin() * 0.3 + 0.7) as f32;
        (pulse * 255.0) as u8
    } else {
        255
    };

    let dot_color = egui::Color32::from_rgba_premultiplied(
        (color.r() as u16 * alpha as u16 / 255) as u8,
        (color.g() as u16 * alpha as u16 / 255) as u8,
        (color.b() as u16 * alpha as u16 / 255) as u8,
        alpha,
    );

    ui.horizontal(|ui| {
        // Draw dot
        let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
        let center = rect.center();
        ui.painter().circle_filled(center, 5.0, dot_color);

        ui.label(
            egui::RichText::new(status.label())
                .size(14.0)
                .color(color),
        );

        if status == AppStatus::Processing {
            ui.spinner();
        }
    });
}
