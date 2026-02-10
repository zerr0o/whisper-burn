//! # Whisper Burn
//!
//! OpenAI Whisper Large V3 / V3 Turbo in pure Rust using the Burn framework.
//! Supports GPU inference via Vulkan/wgpu with Q4_0 quantized weights.
//!
//! ## Supported Models
//!
//! - **Whisper Large V3** (1.55B params): Best accuracy, 32 decoder layers
//! - **Whisper Large V3 Turbo** (809M params): 6x faster, 4 decoder layers
//!
//! ## Architecture
//!
//! ```text
//! Audio (16kHz) -> Mel [1, 128, 3000] -> Encoder [1, 1500, 1280]
//!   -> Decoder (autoregressive) -> Token IDs -> Text
//! ```

pub mod audio;
pub mod gguf;
pub mod model;
pub mod tokenizer;
pub mod transcribe;

#[cfg(feature = "native")]
pub mod native;

pub use audio::AudioBuffer;

/// Language selection for transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Auto,
    French,
    English,
    Spanish,
}

impl Language {
    pub fn display_name(self) -> &'static str {
        match self {
            Language::Auto => "Auto",
            Language::French => "Français",
            Language::English => "English",
            Language::Spanish => "Español",
        }
    }

    pub fn code(self) -> Option<&'static str> {
        match self {
            Language::Auto => None,
            Language::French => Some("fr"),
            Language::English => Some("en"),
            Language::Spanish => Some("es"),
        }
    }

    pub const ALL: [Language; 4] = [
        Language::Auto,
        Language::French,
        Language::English,
        Language::Spanish,
    ];
}
