use super::download::{self, ModelVariant};

pub fn delete_model(variant: ModelVariant) -> Result<(), String> {
    let path = download::gguf_path(variant);
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| format!("Failed to delete model: {e}"))
    } else {
        Ok(())
    }
}

pub fn model_disk_size(variant: ModelVariant) -> Option<u64> {
    let path = download::gguf_path(variant);
    std::fs::metadata(&path).ok().map(|m| m.len())
}

pub fn format_size(bytes: u64) -> String {
    if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
