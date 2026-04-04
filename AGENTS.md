# AGENTS.md

This file gives coding agents a current map of the repository so they can work from the code as it exists today.

## Project Overview

`whisper-burn` is a native Rust Whisper speech-recognition app built on Burn + wgpu with fully local GPU inference from GGUF weights and custom WGSL compute shaders.

Important current state:

- The native desktop app currently exposes **Whisper Medium** and **Whisper Large V3**.
- The repository still contains historical references to **Large V3 Turbo** in comments, crate metadata, and the conversion script, but there is **no native UI/runtime config path for Turbo right now**.
- The app is effectively **Windows-first** today: global hotkey polling and auto-mute use Win32 APIs, while the GUI itself is `eframe/egui`.

## Build And Run

```bash
# Build release app
cargo build --release

# Run the native desktop app
cargo run --release --bin whisper-native

# Build library/runtime pieces only (no GUI)
cargo build --release --no-default-features --features wgpu

# Run tests
cargo test
```

There is also a Windows helper script:

```bat
build.bat
build.bat run
build.bat test
build.bat clean
```

Release builds use `lto = true`, `codegen-units = 1`, and `opt-level = 3`. Debug inference is much slower.

## Runtime Behavior

### App flow

The native app in `src/native/app.rs` runs a state machine with these main screens:

- `CheckModel`
- `ChooseModel`
- `ConfirmDownload`
- `Downloading`
- `LoadingModel`
- `Ready`
- `Recording`
- `Transcribing`
- `ModelManager`

Behavior summary:

- On startup, the app looks for model files next to the executable in `models/`.
- If needed, it downloads `tokenizer.json` plus the selected GGUF model from Hugging Face.
- The model is loaded on a background thread, then wrapped in a dedicated inference worker thread.
- Push-to-talk is handled by polling the configured hotkey with `GetAsyncKeyState`.
- Holding the hotkey starts microphone capture; releasing it stops capture and submits the audio to the inference thread.
- The last transcription is shown in the UI and can optionally be auto-pasted into the foreground app.

### Config and storage

- App config is stored in `dirs::config_dir()/whisper-burn/config.json`.
- Saved settings include language, model variant, hotkey, `auto_paste`, and `auto_mute`.
- Downloaded models live in a `models/` folder beside the executable, not in the config directory.

## Inference Pipeline

The transcription pipeline lives in `src/transcribe.rs`:

Audio -> optional resample to 16 kHz -> pad/truncate to 30s -> log-mel spectrogram -> encoder -> greedy decoder with KV cache -> token filtering -> tokenizer decode -> text

Current implementation details:

- Audio is always normalized to a 30-second Whisper chunk (`480_000` samples at 16 kHz).
- Mel frames are padded/truncated to exactly `3000`.
- `n_mels` depends on the loaded config:
  - Medium: `80`
  - Large V3: `128`
- Encoder output shape depends on the model:
  - Medium: `[1, 1500, 1024]`
  - Large V3: `[1, 1500, 1280]`
- Decoding is greedy and uses a KV cache for autoregressive steps.
- Prompt tokens are processed in one batched pass before token-by-token generation begins.

## Module Structure

- `src/audio/`
  - WAV/audio buffer helpers, FFT-based resampling, STFT + mel filterbank.
- `src/gguf/`
  - GGUF reader, Q4_0 tensor handling, loader, fused q4 matmul op, and `shader.wgsl`.
- `src/model/`
  - Whisper encoder/decoder/attention/layers/config.
  - `config.rs` currently defines **Medium** and **Large V3** configs.
- `src/native/`
  - Desktop app shell, download logic, config persistence, hotkey capture, background inference, model manager, auto-paste, Windows audio mute, and egui UI screens.
- `src/transcribe.rs`
  - End-to-end inference pipeline from raw audio buffer to decoded text.
- `src/tokenizer.rs`
  - Wrapper around `tokenizers::Tokenizer` for Whisper token decoding and language token lookup.

## Key Design Decisions

- **Patched `cubecl-wgpu` runtime**
  - `Cargo.toml` patches `cubecl-wgpu` to `patches/cubecl-wgpu-0.9.0/`.
  - If GPU runtime behavior changes, inspect that patch first.

- **Fused Q4 matmul shader**
  - `src/gguf/shader.wgsl` dequantizes Q4_0 blocks on the fly during matrix multiplication.
  - No intermediate full-precision weight buffer is materialized.

- **Background model loading and inference**
  - Model loading happens on a spawned thread from the UI.
  - Actual transcription work runs on a long-lived inference worker thread communicating by channels.

- **Prompt batching + KV cache**
  - The decoder batches the initial prompt to initialize cache efficiently, then switches to cached autoregressive decoding.

- **Windows-specific interaction features**
  - Global hotkey polling is implemented with `GetAsyncKeyState`.
  - Auto-mute uses `IAudioEndpointVolume`.
  - Auto-paste uses clipboard replacement plus simulated `Ctrl+V`.

## Models

### Native app variants

- `Whisper Medium`
  - Downloaded filename: `whisper-medium-q4.gguf`
  - Config: `WhisperConfig::medium()`
- `Whisper Large V3`
  - Downloaded filename: `whisper-large-v3-q4.gguf`
  - Config: `WhisperConfig::large_v3()`

### Conversion script

`scripts/convert_whisper.py` can convert Hugging Face Whisper checkpoints to Q4_0 GGUF. Example commands:

```bash
python scripts/convert_whisper.py --model openai/whisper-medium --output models/whisper-medium-q4.gguf
python scripts/convert_whisper.py --model openai/whisper-large-v3 --output models/whisper-large-v3-q4.gguf
```

Note:

- The script still defaults to `openai/whisper-large-v3-turbo`.
- That does **not** mean the native app currently wires Turbo up as a selectable runtime model.

Quantization behavior:

- Large 2D weight matrices are quantized to Q4_0.
- Biases, layer norms, embeddings, positional embeddings, and conv weights stay F32.

## Testing

```bash
cargo test
cargo test gguf::tests
cargo test test_q4_matmul_small
```

Current test suite:

- 13 unit tests under `src/gguf/tests.rs`
- Coverage includes GGUF parsing, CPU/GPU Q4 dequantization, q4 matmul at multiple shapes, Q4Linear, Q4FFN, batching, and quantize/dequantize roundtrip checks

## Feature Flags

- `wgpu` (default)
  - Enables the Burn wgpu backend.
- `native` (default)
  - Enables the egui desktop app, microphone capture, downloads, clipboard paste, model manager, and Windows integrations.

## Known Documentation Drift

When updating docs or planning features, prefer the code over older prose:

- `README.md` still mentions tray-related functionality, but there is no tray implementation in `src/native/`.
- Several comments and metadata strings still mention Large V3 Turbo even though the active runtime path is Medium + Large V3.
