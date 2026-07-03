//! Sonora — real-time audio metering for the desktop, written in Rust.
//! Captures system audio (WASAPI loopback) or any input device and shows a
//! spectrogram, spectrum analyzer, LUFS/peak loudness and a stereo goniometer.
//! Inspired by MiniMeters (https://minimeters.app).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod analysis;
mod audio;
mod ui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 680.0])
            .with_min_inner_size([700.0, 460.0])
            .with_title("Sonora"),
        ..Default::default()
    };

    eframe::run_native(
        "Sonora",
        options,
        Box::new(|cc| Ok(Box::new(ui::MetersApp::new(cc)))),
    )
}
