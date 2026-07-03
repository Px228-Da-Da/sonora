//! egui front-end: device selection + four meter panels
//! (spectrogram, spectrum analyzer, loudness/peak, goniometer/correlation).

use eframe::egui;
use egui::{
    Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle,
    TextureOptions, Vec2,
};

use crate::analysis::{Analyzer, Snapshot, SPECTRUM_BINS};
use crate::audio::{self, AudioCapture};

const SPECTRO_W: usize = 512;
const SPECTRO_H: usize = 320;
const F_MIN: f32 = 20.0;
const F_MAX: f32 = 20_000.0;

// dB display range for the spectrogram color map.
const SPECTRO_FLOOR: f32 = -100.0;
const SPECTRO_CEIL: f32 = -20.0;

// Palette
const BG: Color32 = Color32::from_rgb(14, 15, 20);
const PANEL: Color32 = Color32::from_rgb(20, 22, 30);
const GRID: Color32 = Color32::from_rgb(40, 44, 56);
const TEXT_DIM: Color32 = Color32::from_rgb(120, 128, 145);
const ACCENT: Color32 = Color32::from_rgb(90, 200, 250);

pub struct MetersApp {
    devices: Vec<audio::DeviceEntry>,
    selected: usize,
    capture: Option<AudioCapture>,
    analyzer: Analyzer,
    error: Option<String>,

    spectro_img: ColorImage,
    spectro_tex: Option<TextureHandle>,
    row_bins: Vec<f32>,
    mapped_sr: u32,
}

impl MetersApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        let devices = audio::list_devices();
        // Default to the system's default output device, captured via loopback.
        let default_name = audio::default_output_device()
            .map(|d| audio::device_name(&d))
            .unwrap_or_default();
        let selected = devices
            .iter()
            .position(|e| {
                e.kind == audio::SourceKind::Loopback
                    && audio::device_name(&e.device) == default_name
            })
            .unwrap_or(0);

        let mut app = Self {
            devices,
            selected,
            capture: None,
            analyzer: Analyzer::new(48_000),
            error: None,
            spectro_img: ColorImage::filled([SPECTRO_W, SPECTRO_H], BG),
            spectro_tex: None,
            row_bins: Vec::new(),
            mapped_sr: 0,
        };
        app.start_selected();
        app
    }

    fn start_selected(&mut self) {
        self.error = None;
        let Some(entry) = self.devices.get(self.selected) else {
            self.error = Some("Устройства не найдены".to_string());
            return;
        };
        match audio::start_capture(&entry.device, entry.kind) {
            Ok(cap) => {
                self.analyzer = Analyzer::new(cap.sample_rate);
                self.rebuild_row_bins(cap.sample_rate);
                self.capture = Some(cap);
            }
            Err(e) => {
                self.capture = None;
                self.error = Some(e);
            }
        }
    }

    /// Precompute, for each spectrogram row, the (fractional) FFT bin to sample.
    /// Row 0 is the top = high frequency; row H-1 is the bottom = low frequency.
    fn rebuild_row_bins(&mut self, sr: u32) {
        let nyquist = sr as f32 / 2.0;
        let f_max = F_MAX.min(nyquist);
        let bins_per_hz = SPECTRUM_BINS as f32 / nyquist;
        self.row_bins = (0..SPECTRO_H)
            .map(|row| {
                let frac = row as f32 / (SPECTRO_H - 1) as f32; // 0 at top
                let freq = f_max * (F_MIN / f_max).powf(frac);
                freq * bins_per_hz
            })
            .collect();
        self.mapped_sr = sr;
    }

    fn ingest_audio(&mut self) {
        if let Some(cap) = &self.capture {
            let frames = if let Ok(mut s) = cap.shared.lock() {
                s.drain()
            } else {
                Vec::new()
            };
            if !frames.is_empty() {
                self.analyzer.process(&frames);
            }
        }
    }

    fn push_spectro_column(&mut self, spectrum_db: &[f32]) {
        // scroll left by one column
        for y in 0..SPECTRO_H {
            let base = y * SPECTRO_W;
            for x in 0..SPECTRO_W - 1 {
                self.spectro_img.pixels[base + x] = self.spectro_img.pixels[base + x + 1];
            }
        }
        // write newest column at the right edge
        for row in 0..SPECTRO_H {
            let bin_f = self.row_bins.get(row).copied().unwrap_or(0.0);
            let db = sample_spectrum(spectrum_db, bin_f);
            let t = ((db - SPECTRO_FLOOR) / (SPECTRO_CEIL - SPECTRO_FLOOR)).clamp(0.0, 1.0);
            self.spectro_img.pixels[row * SPECTRO_W + (SPECTRO_W - 1)] = magma(t);
        }
    }
}

impl eframe::App for MetersApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.ingest_audio();
        let snap = self.analyzer.snapshot();

        if self.mapped_sr != snap.sample_rate {
            self.rebuild_row_bins(snap.sample_rate);
        }
        self.push_spectro_column(&snap.spectrum_db);

        // upload spectrogram texture
        let tex = self.spectro_tex.get_or_insert_with(|| {
            ctx.load_texture("spectrogram", self.spectro_img.clone(), TextureOptions::LINEAR)
        });
        tex.set(self.spectro_img.clone(), TextureOptions::LINEAR);
        let tex_id = tex.id();

        // ---- top control bar ----
        egui::Panel::top("controls").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Sonora").strong().color(ACCENT));
                ui.separator();
                ui.label("Источник:");
                let current = self
                    .devices
                    .get(self.selected)
                    .map(|e| e.label.clone())
                    .unwrap_or_else(|| "—".to_string());
                let mut new_sel = self.selected;
                egui::ComboBox::from_id_salt("device")
                    .selected_text(current)
                    .width(340.0)
                    .show_ui(ui, |ui| {
                        for (i, entry) in self.devices.iter().enumerate() {
                            ui.selectable_value(&mut new_sel, i, &entry.label);
                        }
                    });
                if new_sel != self.selected {
                    self.selected = new_sel;
                    self.start_selected();
                }
                ui.separator();
                if let Some(cap) = &self.capture {
                    let mode = match self.devices.get(self.selected).map(|e| e.kind) {
                        Some(audio::SourceKind::Input) => "mic",
                        _ => "loopback",
                    };
                    ui.label(
                        egui::RichText::new(format!(
                            "{} Hz · {} ch · {mode}",
                            cap.sample_rate, cap.channels
                        ))
                        .color(TEXT_DIM),
                    );
                }
                if ui.button("Сброс LUFS-I").clicked() {
                    self.analyzer.reset_integrated();
                }
                if let Some(err) = &self.error {
                    ui.colored_label(Color32::from_rgb(240, 90, 90), format!("⚠ {err}"));
                }
            });
        });

        // ---- right panel: loudness, peak, correlation, goniometer ----
        egui::Panel::right("meters")
            .resizable(false)
            .exact_size(300.0)
            .frame(egui::Frame::default().fill(PANEL).inner_margin(10.0))
            .show(ui, |ui| {
                draw_loudness(ui, &snap);
                ui.add_space(8.0);
                draw_peaks(ui, &snap);
                ui.add_space(8.0);
                draw_correlation(ui, &snap);
                ui.add_space(8.0);
                draw_goniometer(ui, &snap);
            });

        // ---- central: spectrogram (top) + spectrum analyzer (bottom) ----
        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(BG))
            .show(ui, |ui| {
                let full = ui.available_rect_before_wrap();
                let split = full.top() + full.height() * 0.6;
                let spectro_rect = Rect::from_min_max(full.min, Pos2::new(full.right(), split));
                let spectrum_rect =
                    Rect::from_min_max(Pos2::new(full.left(), split + 4.0), full.max);

                let painter = ui.painter_at(full);
                painter.image(
                    tex_id,
                    spectro_rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
                draw_spectro_freq_labels(&painter, spectro_rect, snap.sample_rate);
                draw_spectrum(&painter, spectrum_rect, &snap);
            });

        ctx.request_repaint();
    }
}

// ---------- panel drawing ----------

fn draw_loudness(ui: &mut egui::Ui, snap: &Snapshot) {
    ui.label(egui::RichText::new("ГРОМКОСТЬ (LUFS)").strong().color(TEXT_DIM));
    let row = |ui: &mut egui::Ui, label: &str, val: f32, big: bool| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(label).color(TEXT_DIM));
            let txt = if val.is_finite() {
                format!("{val:>7.1}")
            } else {
                "  -∞  ".to_string()
            };
            let size = if big { 22.0 } else { 16.0 };
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(txt)
                        .monospace()
                        .size(size)
                        .color(Color32::from_rgb(230, 235, 245)),
                );
            });
        });
    };
    row(ui, "Momentary", snap.momentary, false);
    row(ui, "Short-term", snap.short_term, false);
    row(ui, "Integrated", snap.integrated, true);

    // momentary bar over -40..0 LUFS
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 14.0), Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 3.0, Color32::from_rgb(30, 33, 44));
    let t = ((snap.momentary + 40.0) / 40.0).clamp(0.0, 1.0);
    let mut fill = rect;
    fill.set_width(rect.width() * t);
    p.rect_filled(fill, 3.0, loudness_color(snap.momentary));
}

fn draw_peaks(ui: &mut egui::Ui, snap: &Snapshot) {
    ui.label(egui::RichText::new("PEAK (dBFS)").strong().color(TEXT_DIM));
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 60.0), Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 4.0, Color32::from_rgb(30, 33, 44));

    let labels = ["L", "R"];
    let h = (rect.height() - 12.0) / 2.0;
    for ch in 0..2 {
        let y0 = rect.top() + 4.0 + ch as f32 * (h + 4.0);
        let bar = Rect::from_min_max(
            Pos2::new(rect.left() + 20.0, y0),
            Pos2::new(rect.right() - 40.0, y0 + h),
        );
        // range -60..0 dB
        let db = snap.peak_db[ch];
        let t = ((db + 60.0) / 60.0).clamp(0.0, 1.0);
        let mut fill = bar;
        fill.set_width(bar.width() * t);
        let col = if db > -1.0 {
            Color32::from_rgb(240, 80, 80)
        } else if db > -6.0 {
            Color32::from_rgb(240, 190, 70)
        } else {
            Color32::from_rgb(90, 200, 120)
        };
        p.rect_filled(fill, 2.0, col);
        // hold marker
        let ht = ((snap.peak_hold_db[ch] + 60.0) / 60.0).clamp(0.0, 1.0);
        let hx = bar.left() + bar.width() * ht;
        p.vline(hx, bar.y_range(), Stroke::new(1.5, Color32::WHITE));
        p.text(
            Pos2::new(rect.left() + 2.0, y0 + h / 2.0),
            Align2::LEFT_CENTER,
            labels[ch],
            FontId::monospace(11.0),
            TEXT_DIM,
        );
        p.text(
            Pos2::new(rect.right() - 2.0, y0 + h / 2.0),
            Align2::RIGHT_CENTER,
            if db.is_finite() { format!("{db:.0}") } else { "-∞".into() },
            FontId::monospace(11.0),
            Color32::from_rgb(210, 215, 225),
        );
    }
}

fn draw_correlation(ui: &mut egui::Ui, snap: &Snapshot) {
    ui.label(egui::RichText::new("КОРРЕЛЯЦИЯ").strong().color(TEXT_DIM));
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 22.0), Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 3.0, Color32::from_rgb(30, 33, 44));
    // center line at 0
    let cx = rect.center().x;
    p.vline(cx, rect.y_range(), Stroke::new(1.0, GRID));
    let val = snap.correlation;
    let x = rect.left() + (val * 0.5 + 0.5) * rect.width();
    let col = if val < 0.0 {
        Color32::from_rgb(240, 100, 90)
    } else {
        Color32::from_rgb(90, 200, 120)
    };
    let marker = Rect::from_min_max(
        Pos2::new(x.min(cx), rect.top() + 2.0),
        Pos2::new(x.max(cx), rect.bottom() - 2.0),
    );
    p.rect_filled(marker, 0.0, col);
    p.text(
        rect.center(),
        Align2::CENTER_CENTER,
        format!("{val:+.2}"),
        FontId::monospace(12.0),
        Color32::WHITE,
    );
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("-1").small().color(TEXT_DIM));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new("+1").small().color(TEXT_DIM));
        });
    });
}

fn draw_goniometer(ui: &mut egui::Ui, snap: &Snapshot) {
    ui.label(egui::RichText::new("ГОНИОМЕТР").strong().color(TEXT_DIM));
    let side = ui.available_width().min(ui.available_height().max(180.0));
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(side), Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 4.0, Color32::from_rgb(12, 13, 18));
    p.rect_stroke(rect, 4.0, Stroke::new(1.0, GRID), StrokeKind::Inside);

    let c = rect.center();
    let half = rect.width() * 0.5;
    // diagonal guides (L / R axes)
    let d = half * 0.92;
    p.line_segment([Pos2::new(c.x - d, c.y - d), Pos2::new(c.x + d, c.y + d)], Stroke::new(1.0, GRID));
    p.line_segment([Pos2::new(c.x - d, c.y + d), Pos2::new(c.x + d, c.y - d)], Stroke::new(1.0, GRID));
    p.text(Pos2::new(c.x - d, c.y - d), Align2::LEFT_TOP, "L", FontId::monospace(11.0), TEXT_DIM);
    p.text(Pos2::new(c.x + d, c.y - d), Align2::RIGHT_TOP, "R", FontId::monospace(11.0), TEXT_DIM);

    let scale = half * 0.9 / std::f32::consts::SQRT_2;
    let dot = Color32::from_rgb(120, 230, 170);
    for &[l, r] in &snap.gonio {
        let mid = (l + r) * std::f32::consts::FRAC_1_SQRT_2;
        let sidev = (l - r) * std::f32::consts::FRAC_1_SQRT_2;
        let px = c.x + sidev * scale;
        let py = c.y - mid * scale;
        p.rect_filled(
            Rect::from_center_size(Pos2::new(px, py), Vec2::splat(1.6)),
            0.0,
            dot,
        );
    }
}

fn draw_spectrum(painter: &egui::Painter, rect: Rect, snap: &Snapshot) {
    painter.rect_filled(rect, 0.0, Color32::from_rgb(10, 11, 15));
    let sr = snap.sample_rate as f32;
    let nyquist = sr / 2.0;
    let f_max = F_MAX.min(nyquist);
    let bins_per_hz = SPECTRUM_BINS as f32 / nyquist;

    // dB grid
    let db_top = 0.0;
    let db_bot = -100.0;
    for &db in &[0.0, -20.0, -40.0, -60.0, -80.0] {
        let y = rect.top() + (db_top - db) / (db_top - db_bot) * rect.height();
        painter.hline(rect.x_range(), y, Stroke::new(1.0, GRID));
        painter.text(
            Pos2::new(rect.left() + 2.0, y),
            Align2::LEFT_BOTTOM,
            format!("{db:.0}"),
            FontId::monospace(10.0),
            TEXT_DIM,
        );
    }
    // frequency grid
    for &f in &[100.0f32, 1000.0, 10000.0] {
        if f > f_max {
            continue;
        }
        let fx = (f / F_MIN).log10() / (f_max / F_MIN).log10();
        let x = rect.left() + fx * rect.width();
        painter.vline(x, rect.y_range(), Stroke::new(1.0, GRID));
        let label = if f >= 1000.0 { format!("{:.0}k", f / 1000.0) } else { format!("{f:.0}") };
        painter.text(
            Pos2::new(x + 2.0, rect.bottom() - 2.0),
            Align2::LEFT_BOTTOM,
            label,
            FontId::monospace(10.0),
            TEXT_DIM,
        );
    }

    // spectrum curve
    let w = rect.width().max(1.0);
    let steps = w.ceil() as usize;
    let mut points: Vec<Pos2> = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let fx = i as f32 / steps as f32;
        let freq = F_MIN * (f_max / F_MIN).powf(fx);
        let bin_f = freq * bins_per_hz;
        let db = sample_spectrum(&snap.spectrum_db, bin_f);
        let ty = ((db_top - db) / (db_top - db_bot)).clamp(0.0, 1.0);
        let x = rect.left() + fx * rect.width();
        let y = rect.top() + ty * rect.height();
        points.push(Pos2::new(x, y));
    }
    // filled area under the curve
    let mut poly = points.clone();
    poly.push(Pos2::new(rect.right(), rect.bottom()));
    poly.push(Pos2::new(rect.left(), rect.bottom()));
    painter.add(egui::Shape::convex_polygon(
        poly,
        Color32::from_rgba_unmultiplied(90, 200, 250, 40),
        Stroke::NONE,
    ));
    painter.add(egui::Shape::line(points, Stroke::new(1.5, ACCENT)));
}

fn draw_spectro_freq_labels(painter: &egui::Painter, rect: Rect, sr: u32) {
    let nyquist = sr as f32 / 2.0;
    let f_max = F_MAX.min(nyquist);
    for &f in &[100.0f32, 1000.0, 10000.0] {
        if f > f_max {
            continue;
        }
        // frac 0 at top (=f_max), 1 at bottom (=F_MIN)
        let frac = (f_max / f).log10() / (f_max / F_MIN).log10();
        let y = rect.top() + frac * rect.height();
        painter.hline(rect.x_range(), y, Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 20)));
        let label = if f >= 1000.0 { format!("{:.0}k", f / 1000.0) } else { format!("{f:.0}") };
        painter.text(
            Pos2::new(rect.left() + 3.0, y),
            Align2::LEFT_BOTTOM,
            label,
            FontId::monospace(10.0),
            Color32::from_rgb(200, 210, 225),
        );
    }
}

// ---------- helpers ----------

/// Linear interpolation of the (already dB) spectrum at a fractional bin index.
fn sample_spectrum(spectrum_db: &[f32], bin_f: f32) -> f32 {
    if spectrum_db.is_empty() {
        return -120.0;
    }
    let b = bin_f.clamp(0.0, (spectrum_db.len() - 1) as f32);
    let i0 = b.floor() as usize;
    let i1 = (i0 + 1).min(spectrum_db.len() - 1);
    let frac = b - i0 as f32;
    spectrum_db[i0] * (1.0 - frac) + spectrum_db[i1] * frac
}

fn loudness_color(lufs: f32) -> Color32 {
    if lufs > -9.0 {
        Color32::from_rgb(240, 90, 80)
    } else if lufs > -16.0 {
        Color32::from_rgb(240, 190, 70)
    } else {
        Color32::from_rgb(90, 200, 120)
    }
}

/// Magma-like color map for t in [0,1].
fn magma(t: f32) -> Color32 {
    const STOPS: [(f32, f32, f32, f32); 8] = [
        (0.0, 0.0, 0.0, 4.0),
        (0.15, 40.0, 11.0, 84.0),
        (0.30, 101.0, 21.0, 110.0),
        (0.45, 159.0, 42.0, 99.0),
        (0.60, 212.0, 72.0, 66.0),
        (0.75, 245.0, 125.0, 21.0),
        (0.90, 250.0, 193.0, 39.0),
        (1.0, 252.0, 255.0, 164.0),
    ];
    let t = t.clamp(0.0, 1.0);
    let mut i = 0;
    while i + 1 < STOPS.len() && t > STOPS[i + 1].0 {
        i += 1;
    }
    let (t0, r0, g0, b0) = STOPS[i];
    let (t1, r1, g1, b1) = STOPS[(i + 1).min(STOPS.len() - 1)];
    let span = (t1 - t0).max(1e-6);
    let f = ((t - t0) / span).clamp(0.0, 1.0);
    Color32::from_rgb(
        (r0 + (r1 - r0) * f) as u8,
        (g0 + (g1 - g0) * f) as u8,
        (b0 + (b1 - b0) * f) as u8,
    )
}
