# Whisper Burn

Native Rust implementation of OpenAI's Whisper speech recognition with GPU acceleration. Push-to-Talk desktop app for Windows with global hotkey, auto-paste, and system tray integration.

Built with [Burn](https://burn.dev) ML framework and wgpu (Vulkan/Metal/DirectX). Models loaded from Q4_0 quantized GGUF files. Inference runs entirely on GPU through custom WGSL compute shaders.

> **⚠️ Known issue:** The "Minimize to tray" feature does not work yet. If you close the window with this option enabled, the app becomes unresponsive and must be killed from the Task Manager. For now, keep the app window open.

## Features

- **Push-to-Talk** — Hold a global hotkey to record, release to transcribe. Works even when the app is not focused. Supports any key combo including modifier-only (e.g. Ctrl+Win).
- **GPU-accelerated** — Custom WGSL compute shaders for fused Q4 dequantization + matrix multiplication. No intermediate buffers.
- **Pure Rust** — No Python, no ONNX, no external ML runtime. Single binary.
- **99+ languages** — All Whisper-supported languages + automatic language detection.
- **Auto-paste** — Transcribed text is automatically pasted into the active application (Ctrl+V simulation with clipboard preservation).
- **Auto-mute** — System audio is muted during recording to avoid feedback, restored afterwards.
- **System tray** — Runs in the background. Show/hide window, start minimized.
- **Model manager** — Download, switch, or delete models from the UI.
- **Persistent config** — Language, hotkey, model, and all settings saved to `%APPDATA%/whisper-burn/config.json`.
- **Dark theme** — Modern dark UI built with egui.

## Models

| Model | Parameters | GGUF Size | Speed | Accuracy |
|-------|-----------|-----------|-------|----------|
| Whisper Medium | 769M | ~604 MB | Fast | Good |
| Whisper Large V3 | 1.55B | ~800 MB | Slower | Best |

Models are automatically downloaded on first launch. Pre-quantized GGUF files are hosted on [HuggingFace](https://huggingface.co/zerr0o/whisper-burn-gguf).

## Requirements

- Windows 10/11
- GPU with Vulkan, DirectX 12, or Metal support
- ~2 GB VRAM (Large V3) / ~1 GB VRAM (Medium)
- Rust 1.75+ (to build from source)

## Quick Start

### Download a release

Download the latest binary from [Releases](https://github.com/zerr0o/whisper-burn/releases) and run `whisper-native.exe`.

### Build from source

```bash
# Clone
git clone https://github.com/zerr0o/whisper-burn.git
cd whisper-burn

# Build (release, optimized with LTO)
cargo build --release

# Run
cargo run --release --bin whisper-native
```

On Windows, you can also use the build script:

```batch
build.bat          :: Build release
build.bat run      :: Build and run
build.bat test     :: Run tests
build.bat clean    :: Clean build
```

### Library only (no GUI)

```bash
cargo build --release --no-default-features --features wgpu
```

## Usage

1. Launch the app — it will prompt you to download a model on first run
2. Press and hold **F2** (or your configured hotkey) to start recording
3. Release **F2** — the audio is transcribed and the text appears in the app
4. With **auto-paste** enabled, the text is automatically pasted into the active window

### Settings

All settings are accessible directly on the main screen:
- **Hotkey** — Click "Change" and press your desired key combo. Supports any combination of Ctrl/Alt/Shift/Win + trigger key (F1-F12, Space, etc.), or modifier-only combos (e.g. Ctrl+Win).
- **Language** — Auto-detect or force a specific language
- **Auto-paste** — Automatically paste transcription into the active app
- **Auto-mute** — Mute system audio during recording
- **Minimize to tray** — Hide to system tray when closing the window

## Architecture

### Inference Pipeline

```
Audio (any sample rate)
  → Resample to 16 kHz
  → Pad/truncate to 30 seconds
  → Mel spectrogram [1, n_mels, 3000]
  → Encoder (conv1d + transformer blocks)
  → Greedy decoder with KV cache
  → Token IDs → BPE decode → Text
```

### Project Structure

```
src/
├── model/           Whisper architecture (encoder, decoder, attention, layers)
├── gguf/            GGUF parser, Q4_0 tensors, fused GPU kernel (shader.wgsl)
├── audio/           Mel spectrogram, WAV I/O, 16kHz resampling
├── native/          Desktop app (eframe/egui)
│   └── ui/          Screens, theme, settings, waveform visualization
├── transcribe.rs    Core inference pipeline
├── tokenizer.rs     BPE tokenizer wrapper
└── lib.rs           Language definitions
```

### Key Design Decisions

- **Fused Q4 matmul shader** — The WGSL shader reads Q4_0 blocks (18 bytes = f16 scale + 16 packed nibbles) and performs dequantization + matrix multiplication in a single kernel pass. No intermediate F32 buffer.
- **Batched prompt processing** — All prompt tokens processed in one forward pass before autoregressive decoding begins.
- **Im2col Conv1D** — 1D convolution uses im2col transformation for efficient GPU matmul.
- **KV cache** — Decoder caches attention keys/values to avoid recomputation across steps.
- **Patched cubecl-wgpu** — Custom patch of CubeCL's wgpu runtime in `patches/`.

## Model Conversion

Convert HuggingFace Whisper models to Q4_0 GGUF:

```bash
pip install torch safetensors transformers numpy

# Medium
python scripts/convert_whisper.py --model openai/whisper-medium --output models/whisper-medium-q4.gguf

# Large V3
python scripts/convert_whisper.py --model openai/whisper-large-v3 --output models/whisper-large-v3-q4.gguf
```

### Quantization

- **Q4_0 format** — 32 values per block, stored as 18 bytes (f16 scale + 16 nibble bytes)
- **Quantized:** 2D weight matrices with dimensions > 256
- **Not quantized (F32):** Embeddings, biases, layer norms, conv weights

## Testing

```bash
cargo test                          # All 13 tests
cargo test gguf::tests              # GGUF module only
cargo test test_q4_matmul_small     # Single test
```

Tests cover Q4_0 dequantization, GGUF v3 parsing, GPU dequant, Q4 matmul kernel (1280x1280), Q4Linear, Q4FFN, batched matmul, and quantize-roundtrip.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `wgpu` | Yes | GPU backend via Burn |
| `native` | Yes | Desktop app with egui, audio capture, hotkey, tray, etc. |

## Dependencies

**Core:** burn, cubecl, rustfft, hound, rubato, tokenizers, half, byteorder

**Native app:** eframe, egui, cpal, tray-icon, arboard, enigo, dirs, ureq, windows (Win32 API for hotkeys and audio mute)

## License

MIT
