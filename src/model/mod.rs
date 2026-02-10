//! Whisper model architecture.
//!
//! Implements the Whisper Large V3 and V3 Turbo encoder-decoder transformer
//! with Q4_0 quantized linear layers.

pub mod attention;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod layers;
pub mod whisper;

pub use config::WhisperConfig;
pub use whisper::WhisperModel;
