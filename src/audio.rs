//! System-audio (WASAPI loopback) capture via cpal.
//!
//! On Windows, building an *input* stream on an *output* device transparently
//! enables loopback recording, so we enumerate output devices and capture from
//! the one the user picks.

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};

/// Roughly 2 s of stereo audio at 48 kHz — cap so a stalled UI can't grow memory.
const MAX_BUFFERED_FRAMES: usize = 48_000 * 2;

/// Shared buffer: audio thread appends frames, UI drains them each repaint.
#[derive(Default)]
pub struct SharedAudio {
    pub frames: Vec<[f32; 2]>,
}

impl SharedAudio {
    /// Take everything captured since the last call.
    pub fn drain(&mut self) -> Vec<[f32; 2]> {
        std::mem::take(&mut self.frames)
    }

    /// Append captured frames, dropping the oldest if the UI has stalled.
    pub fn push(&mut self, frames: &[[f32; 2]]) {
        self.frames.extend_from_slice(frames);
        let len = self.frames.len();
        if len > MAX_BUFFERED_FRAMES {
            self.frames.drain(0..len - MAX_BUFFERED_FRAMES);
        }
    }
}

/// What is actually producing frames. cpal covers devices; per-application capture
/// needs WASAPI process loopback, which cpal does not expose.
/// Held purely to keep the capture alive: dropping either variant stops it.
#[allow(dead_code)]
enum Backend {
    Device(cpal::Stream),
    #[cfg(windows)]
    App(crate::proc_audio::ProcessCapture),
}

pub struct AudioCapture {
    _backend: Backend,
    pub shared: Arc<Mutex<SharedAudio>>,
    /// Last error reported by the stream callback, for the UI to pick up. The
    /// callback can't print: release builds run without a console.
    stream_error: Arc<Mutex<Option<String>>>,
    pub sample_rate: u32,
    pub channels: u16,
    #[allow(dead_code)]
    pub device_name: String,
}

impl AudioCapture {
    /// Take the pending stream error, if the device faulted since the last call.
    pub fn take_error(&self) -> Option<String> {
        self.stream_error.lock().ok().and_then(|mut slot| slot.take())
    }
}

/// How a device is captured.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// Output/render device captured via WASAPI loopback (system audio).
    Loopback,
    /// Input/capture device (microphone, line-in, virtual cable).
    Input,
}

/// One selectable audio source.
pub struct DeviceEntry {
    pub label: String,
    pub device: cpal::Device,
    pub kind: SourceKind,
}

/// Human-readable device name (cpal 0.18 exposes it via `description()`).
pub fn device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "Unknown".to_string())
}

/// List all selectable sources: output devices (loopback) followed by input devices.
pub fn list_devices() -> Vec<DeviceEntry> {
    let host = cpal::default_host();
    let mut out = Vec::new();
    if let Ok(devices) = host.output_devices() {
        for d in devices {
            out.push(DeviceEntry {
                label: format!("{} · loopback", device_name(&d)),
                device: d,
                kind: SourceKind::Loopback,
            });
        }
    }
    if let Ok(devices) = host.input_devices() {
        for d in devices {
            out.push(DeviceEntry {
                label: format!("{} · вход", device_name(&d)),
                device: d,
                kind: SourceKind::Input,
            });
        }
    }
    out
}

pub fn default_output_device() -> Option<cpal::Device> {
    cpal::default_host().default_output_device()
}

pub fn start_capture(device: &cpal::Device, kind: SourceKind) -> Result<AudioCapture, String> {
    let dev_name = device_name(device);

    // `build_input_stream` handles both cases: on an output (render) device it transparently
    // sets the WASAPI loopback flag; on an input device it captures normally. The format must
    // be fetched from the matching config path (render devices reject the input-config path).
    let supported = match kind {
        SourceKind::Loopback => device
            .default_output_config()
            .map_err(|e| format!("default_output_config failed: {e}"))?,
        SourceKind::Input => device
            .default_input_config()
            .map_err(|e| format!("default_input_config failed: {e}"))?,
    };
    let sample_format = supported.sample_format();
    let sample_rate = supported.sample_rate();
    let channels = supported.channels();
    let config: cpal::StreamConfig = supported.into();

    let shared = Arc::new(Mutex::new(SharedAudio::default()));
    let errors: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let (s, e) = (shared.clone(), errors.clone());
    let stream = match sample_format {
        SampleFormat::F32 => build_stream::<f32>(device, &config, channels, s, e),
        SampleFormat::I16 => build_stream::<i16>(device, &config, channels, s, e),
        SampleFormat::U16 => build_stream::<u16>(device, &config, channels, s, e),
        SampleFormat::I32 => build_stream::<i32>(device, &config, channels, s, e),
        other => Err(format!("unsupported sample format: {other:?}")),
    }?;

    stream.play().map_err(|e| format!("stream.play failed: {e}"))?;

    Ok(AudioCapture {
        _backend: Backend::Device(stream),
        shared,
        stream_error: errors,
        sample_rate,
        channels,
        device_name: dev_name,
    })
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: u16,
    shared: Arc<Mutex<SharedAudio>>,
    errors: Arc<Mutex<Option<String>>>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let ch = channels as usize;
    let err_fn = move |e: cpal::Error| {
        if let Ok(mut slot) = errors.lock() {
            *slot = Some(e.to_string());
        }
    };

    let data_fn = move |data: &[T], _: &cpal::InputCallbackInfo| {
        if ch == 0 {
            return;
        }
        let mut frames: Vec<[f32; 2]> = Vec::with_capacity(data.len() / ch + 1);
        for frame in data.chunks_exact(ch) {
            let l = f32::from_sample(frame[0]);
            let r = if ch > 1 { f32::from_sample(frame[1]) } else { l };
            frames.push([l, r]);
        }
        if let Ok(mut s) = shared.lock() {
            s.frames.extend_from_slice(&frames);
            let len = s.frames.len();
            if len > MAX_BUFFERED_FRAMES {
                s.frames.drain(0..len - MAX_BUFFERED_FRAMES);
            }
        }
    };

    device
        .build_input_stream(config.clone(), data_fn, err_fn, None)
        .map_err(|e| format!("build_input_stream failed: {e}"))
}

/// Capture one application instead of a device: everything `pid` and its children
/// play, and nothing else.
#[cfg(windows)]
pub fn start_app_capture(pid: u32, name: &str) -> Result<AudioCapture, String> {
    let shared = Arc::new(Mutex::new(SharedAudio::default()));
    let errors: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let (capture, sample_rate, channels) =
        crate::proc_audio::start(pid, shared.clone(), errors.clone())?;
    Ok(AudioCapture {
        _backend: Backend::App(capture),
        shared,
        stream_error: errors,
        sample_rate,
        channels,
        device_name: name.to_string(),
    })
}
