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

/// App bar, page title and description, shared by the setup and library screens.
pub(super) fn header(
    ui: &mut egui::Ui,
    title: &str,
    subtitle: &str,
    right: impl FnOnce(&mut egui::Ui),
) {
    theme::top_bar(ui, right);
    ui.add_space(18.0);
    theme::page_title(ui, title, subtitle);
    ui.add_space(14.0);
}

/// One-line factual description of a model variant.
pub(super) fn blurb(variant: ModelVariant) -> &'static str {
    match variant {
        ModelVariant::LargeV3 => "1.55B parameters. Best accuracy.",
        ModelVariant::Medium => "769M parameters. Faster, lower VRAM.",
    }
}

/// Model name with an optional badge, description, size line and right-aligned actions.
pub(super) fn model_card(
    ui: &mut egui::Ui,
    variant: ModelVariant,
    badge: Option<(&str, egui::Color32)>,
    detail: &str,
    actions: impl FnOnce(&mut egui::Ui),
) {
    theme::card().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal_top(|ui| {
            let left = (ui.available_width() - 200.0).max(160.0);
            ui.vertical(|ui| {
                ui.set_max_width(left);
                ui.horizontal(|ui| {
                    ui.label(theme::title(variant.display_name(), 14.0));
                    if let Some((text, color)) = badge {
                        theme::badge(ui, text, color);
                    }
                });
                ui.add_space(2.0);
                ui.label(egui::RichText::new(blurb(variant)).color(theme::MUTED));
                ui.label(theme::meta(detail));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), actions);
        });
    });
    ui.add_space(8.0);
}

pub fn draw_choose_model(ui: &mut egui::Ui) -> ChooseAction {
    let mut action = ChooseAction::None;

    header(
        ui,
        "Choose a model",
        "Download a Whisper model. Inference runs locally on your GPU.",
        |ui| {
            if theme::secondary_button(ui, "Quit").clicked() {
                action = ChooseAction::Quit;
            }
        },
    );

    for variant in [ModelVariant::LargeV3, ModelVariant::Medium] {
        let recommended = variant == ModelVariant::LargeV3;
        model_card(
            ui,
            variant,
            recommended.then_some(("Recommended", theme::ACCENT)),
            &format!("{} download", variant.gguf_size_hint()),
            |ui| {
                let clicked = if recommended {
                    theme::primary_button(ui, "Download").clicked()
                } else {
                    theme::secondary_button(ui, "Download").clicked()
                };
                if clicked {
                    action = ChooseAction::Select(variant);
                }
            },
        );
    }

    action
}

pub fn draw_confirm(ui: &mut egui::Ui, variant: ModelVariant) -> ConfirmAction {
    let mut action = ConfirmAction::None;

    header(
        ui,
        "Download required files",
        "Files are saved in the models folder next to the app.",
        |_| {},
    );

    theme::card().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(theme::section("Files"));
        ui.add_space(6.0);
        file_row(ui, variant.gguf_filename(), variant.gguf_size_hint());
        file_row(ui, "tokenizer.json", "~2 MB");
        ui.add_space(6.0);
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
            ui.label(egui::RichText::new(size).monospace().color(theme::MUTED));
        });
    });
}

pub fn draw_progress(ui: &mut egui::Ui, progress: &DownloadProgress, variant: ModelVariant) {
    header(
        ui,
        "Downloading model files",
        "Keep the app open until both downloads finish.",
        |_| {},
    );

    theme::card().show(ui, |ui| {
        ui.set_width(ui.available_width());
        progress_row(
            ui,
            "tokenizer.json",
            progress.tokenizer_bytes.load(Ordering::Relaxed),
            progress.tokenizer_total.load(Ordering::Relaxed),
        );
        ui.add_space(14.0);
        progress_row(
            ui,
            variant.gguf_filename(),
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
            format!("{}  {percent:>3}%", format_bytes(bytes, total)),
        )
    } else if bytes > 0 {
        (0.0, format!("{} downloaded", human_bytes(bytes)))
    } else {
        (0.0, "Waiting...".to_owned())
    };
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).monospace().color(theme::TEXT));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(caption)
                    .monospace()
                    .size(12.0)
                    .color(theme::MUTED),
            );
        });
    });
    ui.add_space(4.0);
    ui.add(
        egui::ProgressBar::new(frac)
            .fill(theme::ACCENT)
            .desired_height(4.0)
            .corner_radius(2),
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
