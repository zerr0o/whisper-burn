use std::sync::mpsc;
use tracing::{info, warn};

use crate::audio::AudioBuffer;
use crate::transcribe::{self, InferenceState};
use crate::Language;

pub enum InferenceRequest {
    Transcribe {
        samples: Vec<f32>,
        sample_rate: u32,
        language: Language,
    },
    Shutdown,
}

pub enum InferenceResponse {
    Result { text: String, inference_ms: u128 },
    Error(String),
}

pub struct InferenceHandle {
    pub tx: mpsc::Sender<InferenceRequest>,
    pub rx: mpsc::Receiver<InferenceResponse>,
}

pub fn spawn_inference_thread(state: InferenceState) -> InferenceHandle {
    let (req_tx, req_rx) = mpsc::channel::<InferenceRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<InferenceResponse>();

    std::thread::spawn(move || {
        info!("Inference thread started");
        loop {
            match req_rx.recv() {
                Ok(InferenceRequest::Shutdown) | Err(_) => {
                    info!("Inference thread shutting down");
                    break;
                }
                Ok(InferenceRequest::Transcribe {
                    samples,
                    sample_rate,
                    language,
                }) => {
                    let audio = AudioBuffer::new(samples, sample_rate);
                    match transcribe::transcribe(&state, audio, language) {
                        Ok((text, inference_ms)) => {
                            if resp_tx
                                .send(InferenceResponse::Result {
                                    text,
                                    inference_ms,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(e) => {
                            warn!("Inference error: {e}");
                            let _ = resp_tx.send(InferenceResponse::Error(e.to_string()));
                        }
                    }
                }
            }
        }
    });

    InferenceHandle {
        tx: req_tx,
        rx: resp_rx,
    }
}
