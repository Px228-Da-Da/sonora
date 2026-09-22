//! Sonora — real-time audio metering for the desktop, written in Rust.
//! Captures system audio (WASAPI loopback) or any input device and shows a
//! spectrogram, spectrum analyzer, LUFS/peak loudness and a stereo goniometer.
//! Inspired by MiniMeters (https://minimeters.app).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod analysis;
mod audio;
#[cfg(windows)]
mod proc_audio;
mod ui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        // There is no main window: the first enabled meter *is* the root viewport,
        // styled like every other widget. Nothing to minimize, so no window can
        // throttle the others.
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([640.0, 260.0])
            .with_min_inner_size([160.0, 110.0])
            .with_decorations(false)
            .with_always_on_top()
            .with_title("Sonora"),
        // Several windows share one GPU: queue two frames ahead so they pipeline
        // instead of each blocking on its own refresh.
        wgpu_options: eframe::WgpuConfiguration {
            surface: eframe::SurfaceConfig::HIGH_THROUGHPUT,
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "Sonora",
        options,
        Box::new(|cc| Ok(Box::new(ui::MetersApp::new(cc)))),
    )
}
