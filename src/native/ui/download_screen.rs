use eframe::egui;
use std::sync::atomic::Ordering;

use super::theme;
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

/// Eyebrow + title + muted subtitle, shared by the setup screens.
pub(super) fn header(ui: &mut egui::Ui, eyebrow: &str, title: &str, subtitle: &str) {
    ui.label(theme::eyebrow(eyebrow).color(theme::ACCENT));
    ui.add_space(4.0);
    ui.label(theme::heading(title));
    ui.add_space(4.0);
    ui.label(egui::RichText::new(subtitle).size(13.5).color(theme::MUTED));
    ui.add_space(20.0);
}

/// One-line factual description of a model variant.
pub(super) fn blurb(variant: ModelVariant) -> &'static str {
    match variant {
        ModelVariant::LargeV3 => "1.55B parameters. Best accuracy.",
        ModelVariant::Medium => "769M parameters. Faster, lower VRAM.",
    }
}

pub fn draw_choose_model(ui: &mut egui::Ui) -> ChooseAction {
    let mut action = ChooseAction::None;

    header(
        ui,
        "WHISPER BURN",
        "Choose a model",
        "Pick a model to download. Everything runs locally on your GPU.",
    );

    for variant in [ModelVariant::LargeV3, ModelVariant::Medium] {
        let recommended = variant == ModelVariant::LargeV3;
        let mut frame = theme::card();
        if recommended {
            frame = frame.stroke(egui::Stroke::new(1.0_f32, theme::ACCENT));
        }
        frame.show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_top(|ui| {
                let left = (ui.available_width() - 130.0).max(160.0);
                ui.vertical(|ui| {
                    ui.set_max_width(left);
                    if recommended {
                        ui.label(theme::eyebrow("RECOMMENDED").color(theme::ACCENT));
                    } else {
                        ui.label(theme::eyebrow("LIGHTWEIGHT").color(theme::DIM));
                    }
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new(variant.display_name())
                            .size(17.0)
                            .strong()
                            .color(theme::TEXT),
                    );
                    ui.add_space(2.0);
                    ui.label(egui::RichText::new(blurb(variant)).color(theme::MUTED));
                    ui.label(
                        egui::RichText::new(format!("Download {}", variant.gguf_size_hint()))
                            .size(12.5)
                            .color(theme::DIM),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    let clicked = if recommended {
                        theme::primary_button(ui, "Download").clicked()
                    } else {
                        theme::secondary_button(ui, "Download").clicked()
                    };
                    if clicked {
                        action = ChooseAction::Select(variant);
                    }
                });
            });
        });
        ui.add_space(12.0);
    }

    ui.add_space(4.0);
    if theme::secondary_button(ui, "Quit").clicked() {
        action = ChooseAction::Quit;
    }

    action
}

pub fn draw_confirm(ui: &mut egui::Ui, variant: ModelVariant) -> ConfirmAction {
    let mut action = ConfirmAction::None;

    header(
        ui,
        "WHISPER BURN",
        "Download required files",
        "These files are needed before the first transcription.",
    );

    theme::card().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(theme::eyebrow("REQUIRED FILES").color(theme::DIM));
        ui.add_space(8.0);
        file_row(ui, variant.gguf_filename(), variant.gguf_size_hint());
        ui.add_space(6.0);
        file_row(ui, "tokenizer.json", "~2 MB");
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if theme::primary_button(ui, "Download").clicked() {
                    action = ConfirmAction::Download;
                }
                if theme::secondary_button(ui, "Back").clicked() {
                    action = ConfirmAction::Back;
                }
            });
        });
    });

    action
}

fn file_row(ui: &mut egui::Ui, name: &str, size: &str) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(name).monospace().color(theme::TEXT));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(size).color(theme::MUTED));
        });
    });
}

pub fn draw_progress(ui: &mut egui::Ui, progress: &DownloadProgress, variant: ModelVariant) {
    header(
        ui,
        "WHISPER BURN",
        "Downloading model files",
        "Keep the app open until both files finish.",
    );

    theme::card().show(ui, |ui| {
        ui.set_width(ui.available_width());
        progress_row(
            ui,
            "Tokenizer (tokenizer.json)",
            progress.tokenizer_bytes.load(Ordering::Relaxed),
            progress.tokenizer_total.load(Ordering::Relaxed),
        );
        ui.add_space(18.0);
        progress_row(
            ui,
            &format!("Model ({})", variant.gguf_filename()),
            progress.gguf_bytes.load(Ordering::Relaxed),
            progress.gguf_total.load(Ordering::Relaxed),
        );
    });
}

fn progress_row(ui: &mut egui::Ui, label: &str, bytes: u64, total: u64) {
    let (frac, caption) = if total > 0 {
        let frac = (bytes as f32 / total as f32).clamp(0.0, 1.0);
        let percent = (bytes as u128 * 100 / total as u128).min(100);
        (
            frac,
            format!("{percent}%  ·  {}", format_bytes(bytes, total)),
        )
    } else if bytes > 0 {
        (0.0, format!("{} downloaded", human_bytes(bytes)))
    } else {
        (0.0, "Waiting...".to_owned())
    };
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(theme::TEXT));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(caption).size(12.5).color(theme::MUTED));
        });
    });
    ui.add_space(6.0);
    ui.add(
        egui::ProgressBar::new(frac)
            .fill(theme::ACCENT)
            .desired_height(8.0),
    );
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
