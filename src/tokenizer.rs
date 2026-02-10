//! Whisper BPE tokenizer wrapper.
//!
//! Wraps the HuggingFace `tokenizers` crate to decode Whisper token IDs.

use anyhow::Result;
use std::path::Path;

pub struct WhisperTokenizer {
    inner: tokenizers::Tokenizer,
}

impl WhisperTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {e}"))
    }

    /// Start of transcript token.
    pub fn sot_token() -> u32 {
        50258
    }

    /// End of transcript token.
    pub fn eot_token() -> u32 {
        50257
    }

    /// Language token for a given language code.
    pub fn lang_token(lang: &str) -> u32 {
        match lang {
            "en" => 50259,
            "zh" => 50260,
            "de" => 50261,
            "es" => 50262,
            "ru" => 50263,
            "ko" => 50264,
            "fr" => 50265,
            "ja" => 50266,
            "pt" => 50267,
            "tr" => 50268,
            "pl" => 50269,
            "ca" => 50270,
            "nl" => 50271,
            "ar" => 50272,
            "sv" => 50273,
            "it" => 50274,
            "id" => 50275,
            "hi" => 50276,
            _ => 50259, // default to English
        }
    }

    /// Transcribe task token.
    pub fn transcribe_token() -> u32 {
        50359
    }

    /// No timestamps token.
    pub fn no_timestamps_token() -> u32 {
        50363
    }
}
