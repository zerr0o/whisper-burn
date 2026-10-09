use eframe::egui;

use super::{download_screen, theme};
use crate::native::download::{self, ModelVariant};
use crate::native::model_manager;

pub enum ModelManagerAction {
    None,
    Back,
    Delete(ModelVariant),
    Switch(ModelVariant),
    Download(ModelVariant),
}

pub fn draw(ui: &mut egui::Ui, current_variant: ModelVariant) -> ModelManagerAction {
    let mut action = ModelManagerAction::None;

    ui.horizontal_top(|ui| {
        ui.vertical(|ui| {
            ui.label(theme::eyebrow("WHISPER BURN").color(theme::ACCENT));
            ui.add_space(4.0);
            ui.label(theme::heading("Model library"));
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("Manage the models stored beside the app.")
                    .size(13.5)
                    .color(theme::MUTED),
            );
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
            if theme::secondary_button(ui, "Back to dictation").clicked() {
                action = ModelManagerAction::Back;
            }
        });
    });
    ui.add_space(20.0);

    for variant in [ModelVariant::LargeV3, ModelVariant::Medium] {
        let is_installed = download::gguf_path(variant).exists();
        let is_current = variant == current_variant;
        let mut frame = theme::card();
        if is_current {
            frame = frame.stroke(egui::Stroke::new(1.0_f32, theme::ACCENT));
        }
        frame.show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_top(|ui| {
                let left = (ui.available_width() - 190.0).max(160.0);
                ui.vertical(|ui| {
                    ui.set_max_width(left);
                    let (status, color) = if is_current {
                        ("ACTIVE", theme::ACCENT)
                    } else if is_installed {
                        ("INSTALLED", theme::GREEN)
                    } else {
                        ("NOT INSTALLED", theme::DIM)
                    };
                    ui.label(theme::eyebrow(status).color(color));
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new(variant.display_name())
                            .size(17.0)
                            .strong()
                            .color(theme::TEXT),
                    );
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new(download_screen::blurb(variant)).color(theme::MUTED),
                    );
                    let size = match model_manager::model_disk_size(variant) {
                        Some(size) => format!("On disk {}", model_manager::format_size(size)),
                        None => format!("Download {}", variant.gguf_size_hint()),
                    };
                    ui.label(egui::RichText::new(size).size(12.5).color(theme::DIM));
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    if is_installed && !is_current {
                        if theme::primary_button(ui, "Use model").clicked() {
                            action = ModelManagerAction::Switch(variant);
                        }
                        let delete =
                            egui::Button::new(egui::RichText::new("Delete").color(theme::MUTED))
                                .fill(egui::Color32::TRANSPARENT)
                                .min_size(egui::vec2(0.0, 34.0));
                        if ui.add(delete).clicked() {
                            action = ModelManagerAction::Delete(variant);
                        }
                    } else if !is_installed && theme::primary_button(ui, "Download").clicked() {
                        action = ModelManagerAction::Download(variant);
                    }
                });
            });
        });
        ui.add_space(12.0);
    }

    action
}
