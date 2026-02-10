use eframe::egui;
use std::sync::atomic::Ordering;

use crate::native::download::{DownloadProgress, ModelVariant};

pub enum ChooseAction {
    Select(ModelVariant),
    Quit,
    None,
}

pub enum ConfirmAction {
    Download,
    Back,
    None,
}

pub fn draw_choose_model(ui: &mut egui::Ui) -> ChooseAction {
    ui.vertical_centered(|ui| {
        ui.add_space(60.0);
        ui.heading("Whisper Burn");
        ui.add_space(20.0);
        ui.label("Choose a model to download:");
        ui.add_space(20.0);

        let mut action = ChooseAction::None;

        ui.group(|ui| {
            ui.set_min_width(400.0);
            ui.vertical(|ui| {
                if ui
                    .button(
                        egui::RichText::new("Whisper Large V3 Turbo (Recommended)")
                            .size(16.0),
                    )
                    .clicked()
                {
                    action = ChooseAction::Select(ModelVariant::LargeV3Turbo);
                }
                ui.label(
                    egui::RichText::new("  809M params | ~420 MB | 6x faster")
                        .size(12.0)
                        .color(egui::Color32::from_gray(140)),
                );
                ui.add_space(12.0);

                if ui
                    .button(egui::RichText::new("Whisper Large V3 (Max Quality)").size(16.0))
                    .clicked()
                {
                    action = ChooseAction::Select(ModelVariant::LargeV3);
                }
                ui.label(
                    egui::RichText::new("  1.55B params | ~800 MB | Best accuracy")
                        .size(12.0)
                        .color(egui::Color32::from_gray(140)),
                );
            });
        });

        ui.add_space(20.0);
        if ui.button("Quit").clicked() {
            action = ChooseAction::Quit;
        }

        action
    })
    .inner
}

pub fn draw_confirm(ui: &mut egui::Ui, variant: ModelVariant) -> ConfirmAction {
    ui.vertical_centered(|ui| {
        ui.add_space(80.0);
        ui.heading("Whisper Burn");
        ui.add_space(20.0);
        ui.label("The model files need to be downloaded.");
        ui.add_space(8.0);
        ui.label("Required files:");
        ui.label(format!(
            "  {}  ({})",
            variant.gguf_filename(),
            variant.gguf_size_hint()
        ));
        ui.label("  tokenizer.json      (~2 MB)");
        ui.add_space(20.0);

        let mut action = ConfirmAction::None;
        ui.horizontal(|ui| {
            if ui
                .button(egui::RichText::new("Download").size(18.0))
                .clicked()
            {
                action = ConfirmAction::Download;
            }
            ui.add_space(16.0);
            if ui.button("Back").clicked() {
                action = ConfirmAction::Back;
            }
        });
        action
    })
    .inner
}

pub fn draw_progress(ui: &mut egui::Ui, progress: &DownloadProgress, variant: ModelVariant) {
    ui.vertical_centered(|ui| {
        ui.add_space(80.0);
        ui.heading("Downloading model files...");
        ui.add_space(30.0);

        // Tokenizer progress
        let tok_bytes = progress.tokenizer_bytes.load(Ordering::Relaxed);
        let tok_total = progress.tokenizer_total.load(Ordering::Relaxed);
        ui.label("Tokenizer (tokenizer.json)");
        if tok_total > 0 {
            let frac = tok_bytes as f32 / tok_total as f32;
            ui.add(egui::ProgressBar::new(frac).text(format_bytes(tok_bytes, tok_total)));
        } else if tok_bytes > 0 {
            ui.add(
                egui::ProgressBar::new(0.0)
                    .text(format!("{} downloaded", human_bytes(tok_bytes))),
            );
        } else {
            ui.add(egui::ProgressBar::new(0.0).text("Waiting..."));
        }

        ui.add_space(16.0);

        // GGUF progress
        let gguf_bytes = progress.gguf_bytes.load(Ordering::Relaxed);
        let gguf_total = progress.gguf_total.load(Ordering::Relaxed);
        ui.label(format!("Model ({})", variant.gguf_filename()));
        if gguf_total > 0 {
            let frac = gguf_bytes as f32 / gguf_total as f32;
            ui.add(egui::ProgressBar::new(frac).text(format_bytes(gguf_bytes, gguf_total)));
        } else if gguf_bytes > 0 {
            ui.add(
                egui::ProgressBar::new(0.0)
                    .text(format!("{} downloaded", human_bytes(gguf_bytes))),
            );
        } else {
            ui.add(egui::ProgressBar::new(0.0).text("Waiting..."));
        }
    });
}

fn format_bytes(current: u64, total: u64) -> String {
    format!("{} / {}", human_bytes(current), human_bytes(total))
}

fn human_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
