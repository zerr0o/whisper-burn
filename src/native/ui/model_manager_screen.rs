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

    download_screen::header(
        ui,
        "Models",
        "Models are stored in the models folder next to the app.",
        |ui| {
            if theme::secondary_button(ui, "Back").clicked() {
                action = ModelManagerAction::Back;
            }
        },
    );

    for variant in [ModelVariant::LargeV3, ModelVariant::Medium] {
        let is_installed = download::gguf_path(variant).exists();
        let is_current = variant == current_variant;
        let badge = if is_current {
            Some(("Active", theme::ACCENT))
        } else if is_installed {
            Some(("Installed", theme::GREEN))
        } else {
            None
        };
        let detail = match model_manager::model_disk_size(variant) {
            Some(size) => format!("{} on disk", model_manager::format_size(size)),
            None => format!("{} download", variant.gguf_size_hint()),
        };
        download_screen::model_card(ui, variant, badge, &detail, |ui| {
            if is_installed && !is_current {
                if theme::primary_button(ui, "Use model").clicked() {
                    action = ModelManagerAction::Switch(variant);
                }
                if theme::quiet_button(ui, "Delete").clicked() {
                    action = ModelManagerAction::Delete(variant);
                }
            } else if !is_installed && theme::primary_button(ui, "Download").clicked() {
                action = ModelManagerAction::Download(variant);
            }
        });
    }

    action
}
