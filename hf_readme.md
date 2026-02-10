---
license: mit
language:
  - multilingual
tags:
  - whisper
  - speech-recognition
  - gguf
  - quantized
  - burn
  - rust
  - wgpu
pipeline_tag: automatic-speech-recognition
---

# Whisper Burn GGUF — Q4_0 Quantized Models

Q4_0 quantized GGUF versions of OpenAI's Whisper models, optimized for GPU inference with [whisper-burn](https://github.com/zerr0o/whisper-burn).

## Files

| File | Model | Size | Parameters |
|------|-------|------|------------|
| `whisper-large-v3-q4.gguf` | Whisper Large V3 | ~1.0 GB | 1550M (32 encoder + 32 decoder layers) |
| `whisper-large-v3-turbo-q4.gguf` | Whisper Large V3 Turbo | ~712 MB | 809M (32 encoder + 4 decoder layers) |
| `whisper-medium-q4.gguf` | Whisper Medium | ~604 MB | 769M (24 encoder + 24 decoder layers) |
| `tokenizer.json` | BPE tokenizer | ~2.1 MB | Shared by all models |

## Quantization Details

- **Format:** GGUF v3 with Q4_0 quantization
- **What's quantized:** 2D weight matrices with dimensions > 256 are quantized to 4-bit (Q4_0 blocks: f16 scale + 16 packed nibble bytes per 32 elements)
- **What stays F32:** Token embeddings, positional embeddings, biases, layer norms, and small matrices
- **Conversion script:** `scripts/convert_whisper.py` from the whisper-burn repository

## Model Comparison

| Model | Mel bins | Hidden dim | Encoder layers | Decoder layers | Accuracy | Speed |
|-------|----------|-----------|----------------|----------------|----------|-------|
| Large V3 | 128 | 1280 | 32 | 32 | Best | Slower |
| Large V3 Turbo | 128 | 1280 | 32 | 4 | Very good | Fast |
| Medium | 80 | 1024 | 24 | 24 | Good | Fastest |

## Usage with whisper-burn

These models are automatically downloaded by the whisper-burn desktop application. You can also download them manually:

```bash
# Download Large V3 Turbo (recommended — fast & accurate)
wget https://huggingface.co/zerr0o/whisper-burn-gguf/resolve/main/whisper-large-v3-turbo-q4.gguf

# Download Large V3 (highest accuracy)
wget https://huggingface.co/zerr0o/whisper-burn-gguf/resolve/main/whisper-large-v3-q4.gguf

# Download Medium (fastest, lower VRAM)
wget https://huggingface.co/zerr0o/whisper-burn-gguf/resolve/main/whisper-medium-q4.gguf

# Download tokenizer (required for all models)
wget https://huggingface.co/zerr0o/whisper-burn-gguf/resolve/main/tokenizer.json
```

Place all files in a `models/` directory next to the whisper-burn executable.

## About whisper-burn

[whisper-burn](https://github.com/zerr0o/whisper-burn) is a native Rust implementation of OpenAI's Whisper using the [Burn](https://burn.dev) ML framework with GPU acceleration via wgpu (Vulkan/Metal/DirectX).

Key features:
- **Pure Rust** — no Python, no ONNX, no external runtime
- **GPU-accelerated** — custom WGSL compute shaders for fused Q4 dequantization + matrix multiplication
- **Push-to-Talk** — global hotkey (F2), system tray, auto-paste, auto-mute
- **99 languages** — all Whisper-supported languages + automatic detection
- **Windows native** — desktop app with dark theme UI

### Inference Pipeline

```
Audio → Resample 16kHz → Mel spectrogram [1, n_mels, 3000]
     → Encoder → Greedy decoder with KV cache
     → Token IDs → Text
```

## Source Models

- [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- [openai/whisper-medium](https://huggingface.co/openai/whisper-medium)

## License

The quantized weights inherit the license from the original OpenAI Whisper models (MIT License).
