# Sonora

Real-time audio metering for the desktop, written in Rust. Sonora captures your
**system audio** (WASAPI loopback on Windows) or any **input device** and shows four
live meters in a single window:

- **Spectrogram** — scrolling time/frequency map (log frequency, magma color map)
- **Spectrum analyzer** — instantaneous FFT magnitude, logarithmic frequency axis
- **Loudness (LUFS)** — Momentary / Short-term / Integrated per ITU-R BS.1770-4
  (K-weighting + gating), plus sample-peak meters with peak-hold
- **Goniometer & correlation** — stereo vectorscope (mid/side) and a phase-correlation meter

Inspired by [MiniMeters](https://minimeters.app). Built for learning and personal use.

## Screenshot

*(add a screenshot here, e.g. `docs/screenshot.png`)*

## Tech stack

- [`eframe`/`egui`](https://github.com/emilk/egui) — window + immediate-mode GUI and custom painting
- [`cpal`](https://github.com/RustAudio/cpal) — cross-platform audio capture (WASAPI loopback on Windows)
- [`rustfft`](https://github.com/ejmahler/RustFFT) — FFT for the spectrum and spectrogram

## Build & run

Requires a recent [Rust toolchain](https://rustup.rs/).

```bash
git clone https://github.com/Px228-Da-Da/sonora
cd sonora
cargo run --release
```

Then start playing any audio and the meters come to life.

## Usage

- **Источник (Source)** dropdown at the top selects the capture device:
  - entries tagged `· loopback` capture a system **output** device (what you hear)
  - entries tagged `· вход` capture an **input** device (microphone, line-in, virtual cable)
- The default selection is your default output device via loopback.
- **Сброс LUFS-I** resets the Integrated loudness measurement and the peak-hold.

## How it works

| Meter | Method |
|-------|--------|
| Spectrum / Spectrogram | 4096-point FFT, Hann window, fast-attack / slow-release smoothing |
| Loudness | Two cascaded biquads for K-weighting (pre-warped to the stream sample rate), 400 ms momentary / 3 s short-term windows, absolute (−70 LUFS) and relative (−10 LU) gating for Integrated |
| Peak | Per-channel sample peak with decay and a peak-hold marker |
| Goniometer | Mid/side projection of recent stereo samples |
| Correlation | Normalized L·R cross-correlation over a ~300 ms window |

The audio callback pushes samples into a shared buffer; the UI thread drains it each
repaint (~60 fps), runs the analysis and renders. See:

- [`src/audio.rs`](src/audio.rs) — device enumeration and capture
- [`src/analysis.rs`](src/analysis.rs) — all DSP
- [`src/ui.rs`](src/ui.rs) — egui front-end and custom drawing

## Platform notes

Developed and tested on **Windows** (WASAPI loopback). `cpal` is cross-platform, so
input-device capture should work on macOS/Linux too; system-audio loopback support
varies by platform and is not yet verified there.

## Roadmap

- True-peak metering (4× oversampling)
- Configurable / draggable tiles like the original
- Persist the selected device and layout between runs
- Screenshot in the README

## License

[MIT](LICENSE)
