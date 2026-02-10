use eframe::egui;

use crate::native::download::ModelVariant;
use crate::native::model_manager;

pub enum ModelManagerAction {
    None,
    Back,
    Delete(ModelVariant),
    Switch(ModelVariant),
    Download(ModelVariant),
}

pub fn draw(
    ui: &mut egui::Ui,
    current_variant: ModelVariant,
) -> ModelManagerAction {
    let mut action = ModelManagerAction::None;

    ui.vertical_centered(|ui| {
        ui.add_space(20.0);
        ui.heading("Model Manager");
        ui.add_space(16.0);

        let installed = model_manager::list_installed_models();
        let installed_variants: Vec<ModelVariant> = installed.iter().map(|m| m.variant).collect();

        for variant in [ModelVariant::LargeV3Turbo, ModelVariant::LargeV3] {
            ui.group(|ui| {
                ui.set_min_width(450.0);
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        let name = variant.display_name();
                        let is_current = variant == current_variant;
                        let is_installed = installed_variants.contains(&variant);

                        ui.label(egui::RichText::new(name).size(16.0).strong());

                        if is_current {
                            ui.colored_label(
                                egui::Color32::from_rgb(80, 200, 120),
                                "Active",
                            );
                        } else if is_installed {
                            ui.colored_label(
                                egui::Color32::from_rgb(80, 140, 230),
                                "Installed",
                            );
                        } else {
                            ui.colored_label(
                                egui::Color32::from_gray(120),
                                "Not installed",
                            );
                        }

                        if let Some(size) = model_manager::model_disk_size(variant) {
                            ui.label(
                                egui::RichText::new(model_manager::format_size(size))
                                    .size(12.0)
                                    .color(egui::Color32::from_gray(120)),
                            );
                        }
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let is_installed = installed_variants.contains(&variant);
                        let is_current = variant == current_variant;

                        if is_installed && !is_current {
                            if ui.button("Delete").clicked() {
                                action = ModelManagerAction::Delete(variant);
                            }
                            if ui.button("Switch").clicked() {
                                action = ModelManagerAction::Switch(variant);
                            }
                        } else if !is_installed {
                            if ui.button("Download").clicked() {
                                action = ModelManagerAction::Download(variant);
                            }
                        }
                    });
                });
            });
            ui.add_space(8.0);
        }

        ui.add_space(16.0);
        if ui.button("Back").clicked() {
            action = ModelManagerAction::Back;
        }
    });

    action
}
