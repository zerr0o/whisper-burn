use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Model variant selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    Medium,
    LargeV3,
}

impl ModelVariant {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Medium => "Whisper Medium",
            Self::LargeV3 => "Whisper Large V3",
        }
    }

    pub fn gguf_filename(&self) -> &'static str {
        match self {
            Self::Medium => "whisper-medium-q4.gguf",
            Self::LargeV3 => "whisper-large-v3-q4.gguf",
        }
    }

    pub fn gguf_size_hint(&self) -> &'static str {
        match self {
            Self::Medium => "~604 MB",
            Self::LargeV3 => "~1.05 GB",
        }
    }

    pub fn gguf_url(&self) -> String {
        format!(
            "https://huggingface.co/zerr0o/whisper-burn-gguf/resolve/main/{}",
            self.gguf_filename()
        )
    }
}

const TOKENIZER_URL: &str =
    "https://huggingface.co/zerr0o/whisper-burn-gguf/resolve/main/tokenizer.json";

#[derive(Default)]
pub struct DownloadProgress {
    pub gguf_bytes: AtomicU64,
    pub gguf_total: AtomicU64,
    pub tokenizer_bytes: AtomicU64,
    pub tokenizer_total: AtomicU64,
    pub error: Mutex<Option<String>>,
    pub done: AtomicBool,
}

pub fn models_dir() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    exe_dir.join("models")
}

pub fn gguf_path(variant: ModelVariant) -> PathBuf {
    models_dir().join(variant.gguf_filename())
}

pub fn tokenizer_path() -> PathBuf {
    models_dir().join("tokenizer.json")
}

pub fn models_present(variant: ModelVariant) -> bool {
    gguf_path(variant).exists() && tokenizer_path().exists()
}

pub fn spawn_download(progress: &Arc<DownloadProgress>, variant: ModelVariant) {
    let p = Arc::clone(progress);

    std::thread::spawn(move || {
        let dir = models_dir();
        if let Err(e) = fs::create_dir_all(&dir) {
            *p.error.lock().unwrap() = Some(format!("Cannot create models dir: {e}"));
            p.done.store(true, Ordering::SeqCst);
            return;
        }

        if let Err(e) = download_file(
            TOKENIZER_URL,
            &tokenizer_path(),
            &p.tokenizer_bytes,
            &p.tokenizer_total,
        ) {
            *p.error.lock().unwrap() = Some(format!("Tokenizer download failed: {e}"));
            p.done.store(true, Ordering::SeqCst);
            return;
        }

        let gguf_url = variant.gguf_url();
        if let Err(e) = download_file(&gguf_url, &gguf_path(variant), &p.gguf_bytes, &p.gguf_total)
        {
            *p.error.lock().unwrap() = Some(format!("GGUF download failed: {e}"));
            p.done.store(true, Ordering::SeqCst);
            return;
        }

        p.done.store(true, Ordering::SeqCst);
    });
}

fn download_file(
    url: &str,
    dest: &Path,
    downloaded: &AtomicU64,
    total: &AtomicU64,
) -> Result<(), String> {
    if dest.exists() {
        let size = fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
        downloaded.store(size, Ordering::SeqCst);
        total.store(size, Ordering::SeqCst);
        return Ok(());
    }

    let resp = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    let content_len: u64 = resp
        .header("content-length")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    total.store(content_len, Ordering::SeqCst);

    let tmp_path = dest.with_extension("tmp");
    let mut file =
        fs::File::create(&tmp_path).map_err(|e| format!("Cannot create temp file: {e}"))?;

    let mut reader = resp.into_reader();
    let mut buf = [0u8; 65536];
    let mut written: u64 = 0;

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("Read error: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("Write error: {e}"))?;
        written += n as u64;
        downloaded.store(written, Ordering::SeqCst);
    }

    file.flush().map_err(|e| format!("Flush error: {e}"))?;
    drop(file);

    fs::rename(&tmp_path, dest).map_err(|e| format!("Rename error: {e}"))?;

    Ok(())
}
