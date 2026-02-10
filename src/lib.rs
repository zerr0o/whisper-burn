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

/// A Whisper language entry.
#[derive(Debug, Clone, Copy)]
pub struct Language {
    pub code: Option<&'static str>,
    pub name: &'static str,
    pub token_id: u32,
}

impl PartialEq for Language {
    fn eq(&self, other: &Self) -> bool {
        self.code == other.code
    }
}
impl Eq for Language {}

impl Language {
    pub fn display_name(self) -> &'static str {
        self.name
    }

    pub fn code(self) -> Option<&'static str> {
        self.code
    }

    pub fn from_code(code: &str) -> &'static Language {
        if code == "auto" {
            return &ALL_LANGUAGES[0];
        }
        ALL_LANGUAGES
            .iter()
            .find(|l| l.code == Some(code))
            .unwrap_or(&ALL_LANGUAGES[0])
    }
}

pub static ALL_LANGUAGES: [Language; 100] = [
    Language { code: None,          name: "Auto",        token_id: 0 },
    Language { code: Some("en"),    name: "English",     token_id: 50259 },
    Language { code: Some("zh"),    name: "Chinese",     token_id: 50260 },
    Language { code: Some("de"),    name: "German",      token_id: 50261 },
    Language { code: Some("es"),    name: "Spanish",     token_id: 50262 },
    Language { code: Some("ru"),    name: "Russian",     token_id: 50263 },
    Language { code: Some("ko"),    name: "Korean",      token_id: 50264 },
    Language { code: Some("fr"),    name: "Français",    token_id: 50265 },
    Language { code: Some("ja"),    name: "Japanese",    token_id: 50266 },
    Language { code: Some("pt"),    name: "Portuguese",  token_id: 50267 },
    Language { code: Some("tr"),    name: "Turkish",     token_id: 50268 },
    Language { code: Some("pl"),    name: "Polish",      token_id: 50269 },
    Language { code: Some("ca"),    name: "Catalan",     token_id: 50270 },
    Language { code: Some("nl"),    name: "Dutch",       token_id: 50271 },
    Language { code: Some("ar"),    name: "Arabic",      token_id: 50272 },
    Language { code: Some("sv"),    name: "Swedish",     token_id: 50273 },
    Language { code: Some("it"),    name: "Italian",     token_id: 50274 },
    Language { code: Some("id"),    name: "Indonesian",  token_id: 50275 },
    Language { code: Some("hi"),    name: "Hindi",       token_id: 50276 },
    Language { code: Some("fi"),    name: "Finnish",     token_id: 50277 },
    Language { code: Some("vi"),    name: "Vietnamese",  token_id: 50278 },
    Language { code: Some("he"),    name: "Hebrew",      token_id: 50279 },
    Language { code: Some("uk"),    name: "Ukrainian",   token_id: 50280 },
    Language { code: Some("el"),    name: "Greek",       token_id: 50281 },
    Language { code: Some("ms"),    name: "Malay",       token_id: 50282 },
    Language { code: Some("cs"),    name: "Czech",       token_id: 50283 },
    Language { code: Some("ro"),    name: "Romanian",    token_id: 50284 },
    Language { code: Some("da"),    name: "Danish",      token_id: 50285 },
    Language { code: Some("hu"),    name: "Hungarian",   token_id: 50286 },
    Language { code: Some("ta"),    name: "Tamil",       token_id: 50287 },
    Language { code: Some("no"),    name: "Norwegian",   token_id: 50288 },
    Language { code: Some("th"),    name: "Thai",        token_id: 50289 },
    Language { code: Some("ur"),    name: "Urdu",        token_id: 50290 },
    Language { code: Some("hr"),    name: "Croatian",    token_id: 50291 },
    Language { code: Some("bg"),    name: "Bulgarian",   token_id: 50292 },
    Language { code: Some("lt"),    name: "Lithuanian",  token_id: 50293 },
    Language { code: Some("la"),    name: "Latin",       token_id: 50294 },
    Language { code: Some("mi"),    name: "Maori",       token_id: 50295 },
    Language { code: Some("ml"),    name: "Malayalam",   token_id: 50296 },
    Language { code: Some("cy"),    name: "Welsh",       token_id: 50297 },
    Language { code: Some("sk"),    name: "Slovak",      token_id: 50298 },
    Language { code: Some("te"),    name: "Telugu",      token_id: 50299 },
    Language { code: Some("fa"),    name: "Persian",     token_id: 50300 },
    Language { code: Some("lv"),    name: "Latvian",     token_id: 50301 },
    Language { code: Some("bn"),    name: "Bengali",     token_id: 50302 },
    Language { code: Some("sr"),    name: "Serbian",     token_id: 50303 },
    Language { code: Some("az"),    name: "Azerbaijani", token_id: 50304 },
    Language { code: Some("sl"),    name: "Slovenian",   token_id: 50305 },
    Language { code: Some("kn"),    name: "Kannada",     token_id: 50306 },
    Language { code: Some("et"),    name: "Estonian",    token_id: 50307 },
    Language { code: Some("mk"),    name: "Macedonian",  token_id: 50308 },
    Language { code: Some("br"),    name: "Breton",      token_id: 50309 },
    Language { code: Some("eu"),    name: "Basque",      token_id: 50310 },
    Language { code: Some("is"),    name: "Icelandic",   token_id: 50311 },
    Language { code: Some("hy"),    name: "Armenian",    token_id: 50312 },
    Language { code: Some("ne"),    name: "Nepali",      token_id: 50313 },
    Language { code: Some("mn"),    name: "Mongolian",   token_id: 50314 },
    Language { code: Some("bs"),    name: "Bosnian",     token_id: 50315 },
    Language { code: Some("kk"),    name: "Kazakh",      token_id: 50316 },
    Language { code: Some("sq"),    name: "Albanian",    token_id: 50317 },
    Language { code: Some("sw"),    name: "Swahili",     token_id: 50318 },
    Language { code: Some("gl"),    name: "Galician",    token_id: 50319 },
    Language { code: Some("mr"),    name: "Marathi",     token_id: 50320 },
    Language { code: Some("pa"),    name: "Punjabi",     token_id: 50321 },
    Language { code: Some("si"),    name: "Sinhala",     token_id: 50322 },
    Language { code: Some("km"),    name: "Khmer",       token_id: 50323 },
    Language { code: Some("sn"),    name: "Shona",       token_id: 50324 },
    Language { code: Some("yo"),    name: "Yoruba",      token_id: 50325 },
    Language { code: Some("so"),    name: "Somali",      token_id: 50326 },
    Language { code: Some("af"),    name: "Afrikaans",   token_id: 50327 },
    Language { code: Some("oc"),    name: "Occitan",     token_id: 50328 },
    Language { code: Some("ka"),    name: "Georgian",    token_id: 50329 },
    Language { code: Some("be"),    name: "Belarusian",  token_id: 50330 },
    Language { code: Some("tg"),    name: "Tajik",       token_id: 50331 },
    Language { code: Some("sd"),    name: "Sindhi",      token_id: 50332 },
    Language { code: Some("gu"),    name: "Gujarati",    token_id: 50333 },
    Language { code: Some("am"),    name: "Amharic",     token_id: 50334 },
    Language { code: Some("yi"),    name: "Yiddish",     token_id: 50335 },
    Language { code: Some("lo"),    name: "Lao",         token_id: 50336 },
    Language { code: Some("uz"),    name: "Uzbek",       token_id: 50337 },
    Language { code: Some("fo"),    name: "Faroese",     token_id: 50338 },
    Language { code: Some("ht"),    name: "Haitian Creole", token_id: 50339 },
    Language { code: Some("ps"),    name: "Pashto",      token_id: 50340 },
    Language { code: Some("tk"),    name: "Turkmen",     token_id: 50341 },
    Language { code: Some("nn"),    name: "Nynorsk",     token_id: 50342 },
    Language { code: Some("mt"),    name: "Maltese",     token_id: 50343 },
    Language { code: Some("sa"),    name: "Sanskrit",    token_id: 50344 },
    Language { code: Some("lb"),    name: "Luxembourgish", token_id: 50345 },
    Language { code: Some("my"),    name: "Myanmar",     token_id: 50346 },
    Language { code: Some("bo"),    name: "Tibetan",     token_id: 50347 },
    Language { code: Some("tl"),    name: "Tagalog",     token_id: 50348 },
    Language { code: Some("mg"),    name: "Malagasy",    token_id: 50349 },
    Language { code: Some("as"),    name: "Assamese",    token_id: 50350 },
    Language { code: Some("tt"),    name: "Tatar",       token_id: 50351 },
    Language { code: Some("haw"),   name: "Hawaiian",    token_id: 50352 },
    Language { code: Some("ln"),    name: "Lingala",     token_id: 50353 },
    Language { code: Some("ha"),    name: "Hausa",       token_id: 50354 },
    Language { code: Some("ba"),    name: "Bashkir",     token_id: 50355 },
    Language { code: Some("jw"),    name: "Javanese",    token_id: 50356 },
    Language { code: Some("su"),    name: "Sundanese",   token_id: 50357 },
];
