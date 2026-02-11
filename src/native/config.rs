use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyConfig {
    pub key: String,
    pub modifiers: Vec<String>,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            key: "F2".into(),
            modifiers: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub language: String,
    pub model_variant: String,
    pub hotkey: HotkeyConfig,
    pub auto_paste: bool,
    pub auto_mute: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            language: "auto".into(),
            model_variant: "large-v3".into(),
            hotkey: HotkeyConfig::default(),
            auto_paste: false,
            auto_mute: false,
        }
    }
}

fn config_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("whisper-burn")
}

fn config_path() -> PathBuf {
    config_dir().join("config.json")
}

pub fn load_config() -> AppConfig {
    let path = config_path();
    if !path.exists() {
        return AppConfig::default();
    }
    match std::fs::read_to_string(&path) {
        Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
        Err(_) => AppConfig::default(),
    }
}

pub fn save_config(config: &AppConfig) {
    let dir = config_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = config_path();
    let tmp = path.with_extension("tmp");
    if let Ok(data) = serde_json::to_string_pretty(config) {
        if std::fs::write(&tmp, &data).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}
