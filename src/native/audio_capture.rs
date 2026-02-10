use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use std::sync::{Arc, Mutex};

pub struct AudioCapture {
    stream: Stream,
    pub buffer: Arc<Mutex<Vec<f32>>>,
    pub sample_rate: u32,
}

impl AudioCapture {
    pub fn start() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("No input device available")?;

        // Try 16kHz mono first, fallback to default config
        let (config, channels) = match device.supported_input_configs() {
            Ok(mut configs) => {
                let mono_16k = configs.find(|c| {
                    c.channels() >= 1
                        && c.min_sample_rate().0 <= 16000
                        && c.max_sample_rate().0 >= 16000
                        && c.sample_format() == SampleFormat::F32
                });
                if let Some(cfg) = mono_16k {
                    let config = cfg.with_sample_rate(cpal::SampleRate(16000)).into();
                    let ch = cfg.channels() as usize;
                    (config, ch)
                } else {
                    let default_cfg = device
                        .default_input_config()
                        .map_err(|e| format!("No input config: {e}"))?;
                    let ch = default_cfg.channels() as usize;
                    (default_cfg.into(), ch)
                }
            }
            Err(_) => {
                let default_cfg = device
                    .default_input_config()
                    .map_err(|e| format!("No input config: {e}"))?;
                let ch = default_cfg.channels() as usize;
                (default_cfg.into(), ch)
            }
        };

        let config: cpal::StreamConfig = config;
        let sample_rate = config.sample_rate.0;
        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let buf_clone = Arc::clone(&buffer);

        let stream = device
            .build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mono: Vec<f32> = if channels > 1 {
                        data.chunks(channels)
                            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
                            .collect()
                    } else {
                        data.to_vec()
                    };
                    if let Ok(mut buf) = buf_clone.lock() {
                        buf.extend_from_slice(&mono);
                    }
                },
                |err| {
                    tracing::warn!("Audio capture error: {err}");
                },
                None,
            )
            .map_err(|e| format!("Failed to build input stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("Failed to start stream: {e}"))?;

        Ok(Self {
            stream,
            buffer,
            sample_rate,
        })
    }

    pub fn take_samples(&self) -> Vec<f32> {
        let mut buf = self.buffer.lock().unwrap();
        std::mem::take(&mut *buf)
    }

    pub fn peek_recent(&self, max_samples: usize) -> Vec<f32> {
        let buf = self.buffer.lock().unwrap();
        if buf.len() <= max_samples {
            buf.clone()
        } else {
            buf[buf.len() - max_samples..].to_vec()
        }
    }

    pub fn stop(self) {
        drop(self.stream);
    }
}
