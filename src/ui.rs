//! egui front-end: device selection + four meter panels
//! (spectrogram, spectrum analyzer, loudness/peak, goniometer/correlation).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use egui::{
    Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle,
    TextureOptions, Vec2,
};

use crate::analysis::{Analyzer, Snapshot, SPECTRUM_BINS};
use crate::audio::{self, AudioCapture};

// Starting size of the spectrogram texture; it is resized to match its on-screen
// size in physical pixels, so it is never magnified (which the linear filter
// would turn into a blur).
const SPECTRO_W0: usize = 512;
const SPECTRO_H0: usize = 320;
const SPECTRO_MIN: usize = 64;
const SPECTRO_MAX: usize = 4096;
/// Time span covered by the full width of the spectrogram. The column rate follows
/// from this and the width, so the scroll speed doesn't change with window size.
///
/// 3.5 s keeps the horizontal scale the original had: it pushed one column per
/// repaint into a 512-wide texture, which on a 144 Hz display worked out to
/// 512 / 144 ≈ 3.5 s — though it stretched to 8.5 s at 60 Hz, since the span was
/// tied to the refresh rate rather than to time.
const SPECTRO_SPAN_SECS: f32 = 3.5;
/// How far the smooth scroll trails the newest column, in columns. Has to exceed the
/// jitter in columns-per-frame, or the display runs past data that doesn't exist yet.
const SCROLL_TARGET_LAG: f64 = 1.5;
/// Share of the remaining error corrected per frame. Only slow drift between the
/// audio and display clocks needs correcting, so this is kept small: the error jumps
/// by a whole column on every write, and any larger share of that reaches the image.
const SCROLL_CORRECTION: f64 = 0.01;
/// Error past which the scroll snaps instead of easing (a stall, or a device change).
const SCROLL_RESYNC_COLUMNS: f64 = 8.0;
/// Repaint interval every viewport asks for. Meters do not need the full refresh
/// rate of a 144 Hz display, and continuous max-rate rendering is both wasted GPU
/// time and what makes the system treat the app like a game.
const FRAME_INTERVAL: Duration = Duration::from_micros(16_667);
/// Ticks closer together than this are skipped: several viewports may each ask the
/// engine to run forward within the same instant.
const MIN_TICK_SECS: f64 = 1.0 / 90.0;
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

/// A meter that can be detached into its own borderless desktop widget.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Meter {
    Spectrogram,
    Spectrum,
    Loudness,
    Goniometer,
}

impl Meter {
    const ALL: [Meter; 4] = [
        Meter::Spectrogram,
        Meter::Spectrum,
        Meter::Loudness,
        Meter::Goniometer,
    ];

    fn index(self) -> usize {
        match self {
            Meter::Spectrogram => 0,
            Meter::Spectrum => 1,
            Meter::Loudness => 2,
            Meter::Goniometer => 3,
        }
    }

    fn title(self) -> &'static str {
        match self {
            Meter::Spectrogram => "Спектрограмма",
            Meter::Spectrum => "Спектр",
            Meter::Loudness => "Громкость",
            Meter::Goniometer => "Гониометр",
        }
    }

    fn default_size(self) -> [f32; 2] {
        match self {
            Meter::Spectrogram => [640.0, 260.0],
            Meter::Spectrum => [640.0, 220.0],
            Meter::Loudness => [300.0, 190.0],
            Meter::Goniometer => [280.0, 340.0],
        }
    }
}

/// State shared by the main window and the detached widgets.
///
/// Widgets are deferred viewports so that minimizing the main window cannot
/// throttle them, and a deferred callback has to be `Send + Sync + 'static`. So
/// everything a widget touches lives here behind a mutex — everything except the
/// cpal stream, which is `!Send` and stays in [`MetersApp`].
struct Engine {
    /// Ring the audio callback writes into; `None` until a device starts.
    audio: Option<Arc<Mutex<audio::SharedAudio>>>,
    analyzer: Analyzer,
    /// Latest analysis, refreshed by [`Engine::tick`].
    snapshot: Snapshot,

    spectro_img: ColorImage,
    spectro_tex: Option<TextureHandle>,
    spectro_size: [usize; 2],
    /// The spectrogram image is a horizontal ring: this is the column holding the
    /// oldest data, which is also where the next column is written. Scrolling by
    /// moving a cursor instead of the pixels keeps the per-frame cost at one column.
    spectro_head: usize,
    /// Size the spectrogram should have, measured from the surface drawing it and
    /// applied on the next tick.
    spectro_target: Option<[usize; 2]>,
    /// Total columns ever written to the ring.
    spectro_written: u64,
    /// Continuous display position, in columns. Advanced by wall-clock time and
    /// pulled gently toward `spectro_written`; see [`Engine::advance_scroll`].
    spectro_scroll: f64,
    /// Sub-pixel offset the spectrogram is drawn at, from the last tick.
    scroll_lag: f32,
    /// Per row, the span of (fractional) FFT bins that row covers.
    row_bands: Vec<(f32, f32)>,
    mapped_sr: u32,
    last_tick: Option<Instant>,
    /// Linear gain the Windows volume control is applying, polled in the background.
    volume: Option<Arc<Mutex<Option<f32>>>>,
    /// Divide that gain back out, so the meters read the level the material actually
    /// has instead of following the system slider. Only meaningful for loopback.
    compensate: bool,
}

/// Everything a window needs to draw one frame, copied out of [`Engine`].
///
/// Windows draw from this, not from the engine itself: holding the lock across
/// rendering lets whichever window repaints most often starve the others out of it.
struct View {
    snapshot: Snapshot,
    tex_id: Option<egui::TextureId>,
    spectro_head: usize,
    spectro_size: [usize; 2],
    scroll_lag: f32,
}

impl Engine {
    fn view(&self) -> View {
        View {
            snapshot: self.snapshot.clone(),
            tex_id: self.tex_id(),
            spectro_head: self.spectro_head,
            spectro_size: self.spectro_size,
            scroll_lag: self.scroll_lag,
        }
    }

    fn new() -> Self {
        let mut analyzer = Analyzer::new(48_000);
        let snapshot = analyzer.snapshot();
        Self {
            audio: None,
            analyzer,
            snapshot,
            spectro_img: ColorImage::filled([SPECTRO_W0, SPECTRO_H0], BG),
            spectro_tex: None,
            spectro_size: [SPECTRO_W0, SPECTRO_H0],
            spectro_head: 0,
            spectro_target: None,
            spectro_written: 0,
            spectro_scroll: 0.0,
            scroll_lag: 0.0,
            row_bands: Vec::new(),
            mapped_sr: 0,
            last_tick: None,
            volume: None,
            compensate: false,
        }
    }

    /// Run the audio pipeline forward.
    ///
    /// Called by every viewport that is about to draw, so the meters keep running at
    /// the rate of whichever window is still being refreshed. Doing the work twice in
    /// one instant would be wasted, so ticks closer together than
    /// `MIN_TICK_SECS` return immediately; everything here is otherwise driven by the
    /// sample clock, so the call rate does not affect the result.
    fn tick(&mut self, ctx: &egui::Context) {
        let now = Instant::now();
        let dt = match self.last_tick {
            Some(t) => {
                let d = (now - t).as_secs_f64();
                if d < MIN_TICK_SECS {
                    return;
                }
                d.min(0.25)
            }
            None => 0.0,
        };
        self.last_tick = Some(now);

        if let Some([w, h]) = self.spectro_target.take() {
            self.resize_spectro(w, h);
        }

        if let Some(ring) = &self.audio {
            let mut frames = match ring.lock() {
                Ok(mut s) => s.drain(),
                Err(_) => Vec::new(),
            };
            if !frames.is_empty() {
                // Loopback is captured after the endpoint volume, so undo it. At or
                // near zero there is nothing left to recover, so leave it alone.
                if let Some(gain) = self.active_gain() {
                    let inv = 1.0 / gain;
                    for f in &mut frames {
                        f[0] *= inv;
                        f[1] *= inv;
                    }
                }
                self.analyzer.process(&frames);
            }
        }

        let snap = self.analyzer.snapshot();
        if self.mapped_sr != snap.sample_rate {
            self.rebuild_row_bands(snap.sample_rate);
        }
        // The texture must exist before new columns can be uploaded into it. A full
        // upload happens only here, on creation and after a resize.
        if self.spectro_tex.is_none() {
            self.spectro_tex = Some(ctx.load_texture(
                "spectrogram",
                self.spectro_img.clone(),
                TextureOptions::LINEAR,
            ));
        }
        self.push_spectro_columns(&snap.columns);
        self.scroll_lag = self.advance_scroll(dt);
        self.snapshot = snap;
    }

    /// The gain to divide out this tick, if compensation is on and readable.
    fn active_gain(&self) -> Option<f32> {
        if !self.compensate {
            return None;
        }
        let g = (*self.volume.as_ref()?.lock().ok()?)?;
        (g > 0.001).then_some(g)
    }

    fn tex_id(&self) -> Option<egui::TextureId> {
        self.spectro_tex.as_ref().map(|t| t.id())
    }

    /// Point the analyzer at a new capture, resetting the analysis for its rate.
    fn attach(&mut self, cap: &AudioCapture) {
        self.analyzer = Analyzer::new(cap.sample_rate);
        self.audio = Some(cap.shared.clone());
        self.sync_column_rate();
        self.rebuild_row_bands(cap.sample_rate);
    }

    fn detach(&mut self) {
        self.audio = None;
    }

    /// Advance the smooth scroll position and return how far behind the newest
    /// column the display should sit, in pixels.
    ///
    /// Columns cannot arrive one per frame: audio comes in ~10 ms packets while the
    /// display refreshes on its own clock, so a frame gets 0, 1 or 2 of them and
    /// stepping the image by whole columns visibly stutters. Instead the position is
    /// driven by elapsed time and only corrected toward the true column count, and
    /// the image is drawn at the resulting sub-pixel offset.
    fn advance_scroll(&mut self, dt: f64) -> f32 {
        let rate = self.spectro_size[0] as f64 / SPECTRO_SPAN_SECS as f64;
        self.spectro_scroll += dt * rate;
        // Aim to trail the newest column slightly, so packet jitter never pushes the
        // display past data that does not exist yet.
        let err = (self.spectro_written as f64 - SCROLL_TARGET_LAG) - self.spectro_scroll;
        if err.abs() > SCROLL_RESYNC_COLUMNS {
            self.spectro_scroll = self.spectro_written as f64 - SCROLL_TARGET_LAG;
        } else {
            self.spectro_scroll += err * SCROLL_CORRECTION;
        }
        (self.spectro_written as f64 - self.spectro_scroll).clamp(0.0, 4.0) as f32
    }

    /// One column per pixel of width, so the full width always spans the same time.
    fn sync_column_rate(&mut self) {
        self.analyzer
            .set_column_rate(self.spectro_size[0] as f32 / SPECTRO_SPAN_SECS);
    }

    /// Precompute, for each spectrogram row, the span of FFT bins it covers.
    /// Row 0 is the top = high frequency; row H-1 is the bottom = low frequency.
    fn rebuild_row_bands(&mut self, sr: u32) {
        let h = self.spectro_size[1];
        let nyquist = (sr as f32 / 2.0).max(1.0);
        let f_max = F_MAX.min(nyquist);
        let bins_per_hz = SPECTRUM_BINS as f32 / nyquist;
        // Row `i` spans the frequencies between edge(i) and edge(i + 1).
        let edge = |i: usize| f_max * (F_MIN / f_max).powf(i as f32 / h as f32);
        self.row_bands = (0..h)
            .map(|row| (edge(row + 1) * bins_per_hz, edge(row) * bins_per_hz))
            .collect();
        self.mapped_sr = sr;
    }

    /// Resize the spectrogram to `w`×`h` physical pixels, carrying the history over:
    /// the newest columns are kept, and rows sit at the same normalized position on
    /// the log-frequency axis at any height, so a vertical resample preserves them.
    fn resize_spectro(&mut self, w: usize, h: usize) {
        let [ow, oh] = self.spectro_size;
        if [w, h] == [ow, oh] {
            return;
        }
        // Unroll the ring into display order (oldest left, newest right) at the new
        // size, which also resets the cursor to 0.
        let mut img = ColorImage::filled([w, h], BG);
        let keep = w.min(ow);
        for y in 0..h {
            let sy = y * oh / h;
            for i in 0..keep {
                let sx = (self.spectro_head + ow - keep + i) % ow;
                img.pixels[y * w + (w - keep) + i] = self.spectro_img.pixels[sy * ow + sx];
            }
        }
        self.spectro_img = img;
        self.spectro_size = [w, h];
        self.spectro_head = 0;
        // The handle is bound to the old size; drop it so it is recreated.
        self.spectro_tex = None;
        if h != oh {
            self.rebuild_row_bands(self.mapped_sr);
        }
        if w != ow {
            self.sync_column_rate();
        }
        // The column rate just changed; don't ease across the discontinuity.
        self.spectro_scroll = self.spectro_written as f64 - SCROLL_TARGET_LAG;
    }

    /// Append every column produced since the last tick, so the time axis stays
    /// linear even when a window drops frames or runs faster than the analysis hop.
    ///
    /// Each column goes to the cursor and is uploaded on its own; a full re-upload
    /// would be megabytes per frame once the texture matches the panel size.
    fn push_spectro_columns(&mut self, columns: &[Vec<f32>]) {
        let [w, h] = self.spectro_size;
        let Some(tex) = &mut self.spectro_tex else {
            return;
        };
        if columns.is_empty() || self.row_bands.len() != h {
            return;
        }
        // Anything older than a full width would be overwritten again anyway.
        let skip = columns.len().saturating_sub(w);
        let mut patch = ColorImage::filled([1, h], BG);
        for col in &columns[skip..] {
            let x = self.spectro_head;
            for row in 0..h {
                let (lo, hi) = self.row_bands[row];
                let db = sample_band(col, lo, hi);
                let t = ((db - SPECTRO_FLOOR) / (SPECTRO_CEIL - SPECTRO_FLOOR)).clamp(0.0, 1.0);
                let px = magma(t);
                patch.pixels[row] = px;
                self.spectro_img.pixels[row * w + x] = px;
            }
            tex.set_partial([x, 0], patch.clone(), TextureOptions::LINEAR);
            self.spectro_head = (x + 1) % w;
            self.spectro_written += 1;
        }
    }

}

/// Paint the spectrogram into `rect`, returning the texture size that surface wants
/// — the main window and the detached widget are different sizes, and whichever is
/// showing it sizes it. Applied on the next tick: the rect is only known here.
fn draw_spectro_surface(
    painter: &egui::Painter,
    rect: Rect,
    view: &View,
    ppp: f32,
) -> Option<[usize; 2]> {
    let want = [
        ((rect.width() * ppp).round() as usize).clamp(SPECTRO_MIN, SPECTRO_MAX),
        ((rect.height() * ppp).round() as usize).clamp(SPECTRO_MIN, SPECTRO_MAX),
    ];
    if let Some(tex_id) = view.tex_id {
        draw_spectrogram(
            painter,
            rect,
            tex_id,
            view.spectro_head,
            view.spectro_size[0],
            view.scroll_lag,
        );
    }
    draw_spectro_freq_labels(painter, rect, view.snapshot.sample_rate);
    (want != view.spectro_size).then_some(want)
}

/// Something the context menu asked for, applied after drawing so the menu closure
/// doesn't need to borrow the whole app.
enum Action {
    SetDevice(usize),
    /// Capture one application by pid instead of a device.
    SetApp(u32),
    ResetLufs,
    ToggleCompensate,
    Toggle(usize),
    Quit,
}

/// The read-only bits the context menu shows.
/// What the capture is currently pointed at.
#[derive(Clone, PartialEq)]
enum Source {
    Device(usize),
    App(u32),
}

struct MenuData {
    devices: Vec<String>,
    apps: Vec<(u32, String)>,
    source: Source,
    enabled: [bool; 4],
    status: String,
    error: Option<String>,
    compensate: bool,
    /// Gain the system slider is applying right now, for the readout.
    gain: Option<f32>,
}

pub struct MetersApp {
    devices: Vec<audio::DeviceEntry>,
    source: Source,
    /// Applications holding an audio session, refreshed on a background thread.
    apps: Arc<Mutex<Vec<crate::proc_audio::AppEntry>>>,
    /// The capture backend. A cpal stream is `!Send`, so it never enters [`Engine`];
    /// only the ring it writes into is shared.
    capture: Option<AudioCapture>,
    engine: Arc<Mutex<Engine>>,
    error: Option<String>,
    /// Which meters are on screen, indexed by [`Meter::index`].
    enabled: [bool; 4],
    /// Read the Windows volume back out of loopback levels.
    compensate: bool,
    volume: Arc<Mutex<Option<f32>>>,
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
            source: Source::Device(selected),
            apps: crate::proc_audio::spawn_app_watcher(),
            capture: None,
            engine: Arc::new(Mutex::new(Engine::new())),
            error: None,
            enabled: [true; 4],
            compensate: true,
            volume: crate::proc_audio::spawn_volume_watcher(),
        };
        if let Ok(mut eng) = app.engine.lock() {
            eng.volume = Some(app.volume.clone());
        }
        app.start_selected();
        app
    }

    fn start_selected(&mut self) {
        self.error = None;
        let started = match &self.source {
            Source::Device(i) => match self.devices.get(*i) {
                Some(entry) => audio::start_capture(&entry.device, entry.kind),
                None => Err("Устройства не найдены".to_string()),
            },
            Source::App(pid) => {
                let name = self
                    .apps
                    .lock()
                    .ok()
                    .and_then(|list| {
                        list.iter().find(|a| a.pid == *pid).map(|a| a.name.clone())
                    })
                    .unwrap_or_else(|| format!("pid {pid}"));
                audio::start_app_capture(*pid, &name)
            }
        };
        // Only shared-mode loopback of an output device carries the endpoint volume.
        // A microphone does not, and per-application capture taps the app's own
        // stream ahead of the endpoint mix.
        let is_loopback = matches!(&self.source, Source::Device(i)
            if self.devices.get(*i).map(|e| e.kind) == Some(audio::SourceKind::Loopback));
        match started {
            Ok(cap) => {
                if let Ok(mut eng) = self.engine.lock() {
                    eng.attach(&cap);
                    eng.compensate = self.compensate && is_loopback;
                }
                self.capture = Some(cap);
            }
            Err(e) => {
                self.capture = None;
                if let Ok(mut eng) = self.engine.lock() {
                    eng.detach();
                }
                self.error = Some(e);
            }
        }
    }

    /// Surface a fault reported by the stream callback (device unplugged, format
    /// change) instead of leaving the meters silently frozen.
    fn poll_stream_error(&mut self) {
        if let Some(msg) = self.capture.as_ref().and_then(|c| c.take_error()) {
            self.error = Some(format!("поток: {msg}"));
        }
    }

    fn menu_data(&self) -> MenuData {
        MenuData {
            devices: self.devices.iter().map(|d| d.label.clone()).collect(),
            apps: self
                .apps
                .lock()
                .map(|l| l.iter().map(|a| (a.pid, a.name.clone())).collect())
                .unwrap_or_default(),
            source: self.source.clone(),
            enabled: self.enabled,
            status: match &self.capture {
                Some(c) => format!("{} Hz · {} ch", c.sample_rate, c.channels),
                None => "нет захвата".to_string(),
            },
            error: self.error.clone(),
            compensate: self.compensate,
            gain: self.volume.lock().ok().and_then(|g| *g),
        }
    }

    fn apply(&mut self, actions: Vec<Action>, ctx: &egui::Context) {
        for a in actions {
            match a {
                Action::SetDevice(i) => {
                    if self.source != Source::Device(i) {
                        self.source = Source::Device(i);
                        self.start_selected();
                    }
                }
                Action::SetApp(pid) => {
                    if self.source != Source::App(pid) {
                        self.source = Source::App(pid);
                        self.start_selected();
                    }
                }
                Action::ToggleCompensate => {
                    self.compensate = !self.compensate;
                    self.start_selected();
                }
                Action::ResetLufs => {
                    if let Ok(mut eng) = self.engine.lock() {
                        eng.analyzer.reset_integrated();
                    }
                }
                Action::Toggle(i) => self.enabled[i] = !self.enabled[i],
                Action::Quit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            }
        }
    }
}

fn widget_frame() -> egui::Frame {
    egui::Frame::default()
        .fill(PANEL)
        .stroke(Stroke::new(1.0, GRID))
        .inner_margin(6.0)
}

impl eframe::App for MetersApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_stream_error();

        // Tick, copy, release: the lock is never held across drawing.
        let view = match self.engine.lock() {
            Ok(mut eng) => {
                eng.tick(&ctx);
                eng.view()
            }
            Err(_) => return,
        };

        let menu = self.menu_data();
        let mut actions = Vec::new();
        let mut want_spectro = None;

        // The app has no separate main window: the first enabled meter *is* the root
        // window, the rest are child viewports. Nothing to minimize, so nothing can
        // throttle the others.
        let mut on: Vec<Meter> = Meter::ALL
            .into_iter()
            .filter(|m| self.enabled[m.index()])
            .collect();
        let root_meter = if on.is_empty() { None } else { Some(on.remove(0)) };

        egui::CentralPanel::default()
            .frame(widget_frame())
            .show(ui, |ui| match root_meter {
                Some(meter) => {
                    if let Some(w) = draw_meter(ui, meter, &view, &menu, &mut actions) {
                        want_spectro = Some(w);
                    }
                }
                None => draw_placeholder(ui, &menu, &mut actions),
            });

        for meter in on {
            let mut w = None;
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of(meter.title()),
                egui::ViewportBuilder::default()
                    .with_title(format!("Sonora — {}", meter.title()))
                    .with_decorations(false)
                    .with_always_on_top()
                    .with_taskbar(false)
                    .with_inner_size(meter.default_size())
                    .with_min_inner_size([160.0, 110.0]),
                |ui, _class| {
                    egui::CentralPanel::default()
                        .frame(widget_frame())
                        .show(ui, |ui| {
                            w = draw_meter(ui, meter, &view, &menu, &mut actions);
                        });
                    if ui.ctx().input(|i| i.viewport().close_requested()) {
                        actions.push(Action::Toggle(meter.index()));
                    }
                },
            );
            if w.is_some() {
                want_spectro = w;
            }
        }

        if want_spectro.is_some() {
            if let Ok(mut eng) = self.engine.lock() {
                eng.spectro_target = want_spectro;
            }
        }
        self.apply(actions, &ctx);

        ctx.request_repaint_after(FRAME_INTERVAL);
    }
}

/// Shown when every meter has been switched off, so the app stays reachable: without
/// window decorations the context menu is the only way back.
fn draw_placeholder(ui: &mut egui::Ui, menu: &MenuData, actions: &mut Vec<Action>) {
    let resp = ui.interact(
        ui.max_rect(),
        ui.id().with("empty"),
        Sense::click_and_drag(),
    );
    ui.centered_and_justified(|ui| {
        ui.label(
            egui::RichText::new("Sonora — правый клик для меню")
                .small()
                .color(TEXT_DIM),
        );
    });
    context_menu(&resp, menu, actions);
    if resp.drag_started() {
        ui.ctx()
            .send_viewport_cmd(egui::ViewportCommand::StartDrag);
    }
}

/// Reachable two ways — the "меню" button in every title strip, and right-click
/// anywhere on a widget.
fn context_menu(resp: &egui::Response, menu: &MenuData, actions: &mut Vec<Action>) {
    resp.context_menu(|ui| menu_contents(ui, menu, actions));
}

/// Every control lives here: with no decorations and no main window, this menu is
/// the whole interface.
fn menu_contents(ui: &mut egui::Ui, menu: &MenuData, actions: &mut Vec<Action>) {
    ui.set_min_width(260.0);
    ui.label(egui::RichText::new(&menu.status).small().color(TEXT_DIM));
    if let Some(err) = &menu.error {
        ui.colored_label(Color32::from_rgb(240, 90, 90), format!("⚠ {err}"));
    }
    ui.separator();

    ui.menu_button("Устройство", |ui| {
        for (i, label) in menu.devices.iter().enumerate() {
            if ui.radio(menu.source == Source::Device(i), label).clicked() {
                actions.push(Action::SetDevice(i));
                ui.close();
            }
        }
    });
    ui.menu_button("Программа", |ui| {
        if menu.apps.is_empty() {
            ui.label(
                egui::RichText::new("Нет приложений со звуком")
                    .small()
                    .color(TEXT_DIM),
            );
        }
        for (pid, name) in &menu.apps {
            if ui
                .radio(menu.source == Source::App(*pid), name)
                .on_hover_text(format!("pid {pid} — вместе с дочерними процессами"))
                .clicked()
            {
                actions.push(Action::SetApp(*pid));
                ui.close();
            }
        }
    });
    ui.separator();

    ui.label(egui::RichText::new("Показывать").small().color(TEXT_DIM));
    for m in Meter::ALL {
        if ui.checkbox(&mut menu.enabled[m.index()].clone(), m.title()).clicked() {
            actions.push(Action::Toggle(m.index()));
        }
    }
    ui.separator();

    let mut comp = menu.compensate;
    let label = match menu.gain {
        Some(g) if g > 0.0 => format!("Не зависеть от громкости Windows ({:+.1} dB)", 20.0 * g.log10()),
        Some(_) => "Не зависеть от громкости Windows (звук выключен)".to_string(),
        None => "Не зависеть от громкости Windows".to_string(),
    };
    if ui
        .checkbox(&mut comp, label)
        .on_hover_text(
            "Системный ползунок громкости попадает в loopback-запись.              С этой галкой его влияние вычитается, и метры показывают              уровень самого материала.",
        )
        .clicked()
    {
        actions.push(Action::ToggleCompensate);
    }
    ui.separator();

    if ui.button("Сброс LUFS-I").clicked() {
        actions.push(Action::ResetLufs);
        ui.close();
    }
    if ui.button("Выход").clicked() {
        actions.push(Action::Quit);
    }
}

/// One meter window: a title strip, the meter itself, and the handles that stand in
/// for the window chrome we turned off. Returns the texture size the spectrogram
/// wants, when this is the window showing it.
fn draw_meter(
    ui: &mut egui::Ui,
    meter: Meter,
    view: &View,
    menu: &MenuData,
    actions: &mut Vec<Action>,
) -> Option<[usize; 2]> {
    let ctx = ui.ctx().clone();
    let outer = ui.max_rect();

    // Claimed first, so the close button and the grip below sit on top of it and win
    // the click. Without decorations this is the only way to move the window.
    let body_drag = ui.interact(outer, ui.id().with("drag"), Sense::click_and_drag());

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(meter.title()).small().color(TEXT_DIM));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("×").on_hover_text("Скрыть").clicked() {
                actions.push(Action::Toggle(meter.index()));
            }
            // The only visible entry point to the settings: without decorations
            // there is no menu bar, and right-click alone is too easy to miss.
            ui.menu_button("меню", |ui| menu_contents(ui, menu, actions))
                .response
                .on_hover_text("Источник звука и настройки");
        });
    });

    let rect = ui.available_rect_before_wrap();
    let mut want = None;
    match meter {
        Meter::Spectrogram => {
            let painter = ui.painter_at(rect);
            want = draw_spectro_surface(&painter, rect, view, ctx.pixels_per_point());
        }
        Meter::Spectrum => draw_spectrum(&ui.painter_at(rect), rect, &view.snapshot),
        Meter::Loudness => {
            draw_loudness(ui, &view.snapshot);
            ui.add_space(8.0);
            draw_peaks(ui, &view.snapshot);
        }
        Meter::Goniometer => {
            draw_correlation(ui, &view.snapshot);
            ui.add_space(8.0);
            draw_goniometer(ui, &view.snapshot);
        }
    }

    // Resize grip, last so it takes precedence over the body drag.
    let grip = Rect::from_min_max(outer.max - Vec2::splat(14.0), outer.max);
    let grip_resp = ui.interact(grip, ui.id().with("grip"), Sense::click_and_drag());
    let p = ui.painter();
    for i in 1..4 {
        let d = i as f32 * 4.0;
        p.line_segment(
            [
                Pos2::new(outer.right() - d, outer.bottom()),
                Pos2::new(outer.right(), outer.bottom() - d),
            ],
            Stroke::new(1.0, TEXT_DIM),
        );
    }

    context_menu(&body_drag, menu, actions);

    if grip_resp.drag_started() {
        ctx.send_viewport_cmd(egui::ViewportCommand::BeginResize(
            egui::ResizeDirection::SouthEast,
        ));
    } else if body_drag.drag_started() {
        ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
    }
    want
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
        // Peak over the pixel's own frequency span: past a few kHz one pixel covers
        // dozens of bins, and point-sampling makes narrow tones flicker or vanish.
        let span = |off: f32| F_MIN * (f_max / F_MIN).powf(fx + off / steps as f32);
        let db = sample_band(
            &snap.spectrum_db,
            span(-0.5) * bins_per_hz,
            span(0.5) * bins_per_hz,
        );
        let ty = ((db_top - db) / (db_top - db_bot)).clamp(0.0, 1.0);
        let x = rect.left() + fx * rect.width();
        let y = rect.top() + ty * rect.height();
        points.push(Pos2::new(x, y));
    }
    // Filled area under the curve, as a strip of quads. `convex_polygon` cannot be
    // used: a spectrum outline is deeply concave, and it tessellates that into a fan
    // of overlapping wedges instead of the area under the curve.
    let fill = Color32::from_rgba_unmultiplied(90, 200, 250, 40);
    let mut mesh = egui::Mesh::default();
    for (i, p) in points.iter().enumerate() {
        mesh.colored_vertex(*p, fill);
        mesh.colored_vertex(Pos2::new(p.x, rect.bottom()), fill);
        if i > 0 {
            let base = (i as u32 - 1) * 2;
            mesh.add_triangle(base, base + 1, base + 2);
            mesh.add_triangle(base + 1, base + 2, base + 3);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
    painter.add(egui::Shape::line(points, Stroke::new(1.5, ACCENT)));
}

/// Draw the ring-buffered spectrogram as two quads: the columns from `head` to the
/// end of the texture are the older half and go on the left, the rest follows.
///
/// `lag` shifts the whole image right by a fraction of a column, which is what makes
/// the scroll continuous between the discrete arrivals of new columns. The overhang
/// past the right edge is clipped by the painter.
fn draw_spectrogram(
    painter: &egui::Painter,
    rect: Rect,
    tex_id: egui::TextureId,
    head: usize,
    width: usize,
    lag: f32,
) {
    let u = head as f32 / width as f32;
    let dx = lag * rect.width() / width as f32;
    let split = rect.left() + rect.width() * (1.0 - u) + dx;
    painter.image(
        tex_id,
        Rect::from_min_max(
            Pos2::new(rect.left() + dx, rect.top()),
            Pos2::new(split, rect.bottom()),
        ),
        Rect::from_min_max(Pos2::new(u, 0.0), Pos2::new(1.0, 1.0)),
        Color32::WHITE,
    );
    if head > 0 {
        painter.image(
            tex_id,
            Rect::from_min_max(
                Pos2::new(split, rect.top()),
                Pos2::new(rect.right() + dx, rect.bottom()),
            ),
            Rect::from_min_max(Pos2::ZERO, Pos2::new(u, 1.0)),
            Color32::WHITE,
        );
    }
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

/// Value of the spectrum over the fractional bin span `lo..hi`.
///
/// Spans wider than a bin take the peak, so a narrow tone stays visible where one
/// row or pixel covers many bins. Narrower spans — the low-frequency end, where
/// several rows share one bin — interpolate instead of stair-stepping.
fn sample_band(spectrum_db: &[f32], lo: f32, hi: f32) -> f32 {
    if spectrum_db.is_empty() {
        return -120.0;
    }
    let last = (spectrum_db.len() - 1) as f32;
    let lo = lo.clamp(0.0, last);
    let hi = hi.clamp(lo, last);
    if hi - lo < 1.0 {
        return sample_spectrum(spectrum_db, 0.5 * (lo + hi));
    }
    let i0 = lo.floor() as usize;
    let i1 = (hi.ceil() as usize).min(spectrum_db.len() - 1);
    spectrum_db[i0..=i1]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::SPECTRUM_BINS;

    /// A 19 kHz tone occupies ~2 bins out of 2048. With one row of a 320-row
    /// spectrogram spanning ~37 bins up there, point-sampling misses it almost
    /// always; taking the band peak must always find it.
    #[test]
    fn narrow_high_tone_is_not_missed() {
        let sr = 48_000.0f32;
        let nyquist = sr / 2.0;
        let bins_per_hz = SPECTRUM_BINS as f32 / nyquist;

        let mut spec = vec![-120.0f32; SPECTRUM_BINS];
        let tone_bin = (19_000.0 * bins_per_hz).round() as usize;
        spec[tone_bin] = -10.0;

        let h = 320usize;
        let f_max = F_MAX.min(nyquist);
        let edge = |i: usize| f_max * (F_MIN / f_max).powf(i as f32 / h as f32);

        let mut found_band = false;
        let mut found_point = false;
        for row in 0..h {
            let (lo, hi) = (edge(row + 1) * bins_per_hz, edge(row) * bins_per_hz);
            if sample_band(&spec, lo, hi) > -60.0 {
                found_band = true;
            }
            // what the old code did: one interpolated bin per row
            let frac = row as f32 / (h - 1) as f32;
            let bin_f = f_max * (F_MIN / f_max).powf(frac) * bins_per_hz;
            if sample_spectrum(&spec, bin_f) > -60.0 {
                found_point = true;
            }
        }
        assert!(found_band, "band sampling lost the 19 kHz tone");
        assert!(!found_point, "point sampling unexpectedly found it — test is not proving anything");
    }

    /// Below a bin's width the band must interpolate, so the low end stays smooth
    /// rather than stair-stepping across the rows that share one bin.
    #[test]
    fn narrow_bands_interpolate() {
        let spec = vec![-40.0, -20.0, -60.0];
        assert!((sample_band(&spec, 0.0, 0.5) - -35.0).abs() < 1e-3);
        assert!((sample_band(&spec, 0.5, 0.9) - -26.0).abs() < 1e-3);
        // wide enough to span bins: peak wins
        assert!((sample_band(&spec, 0.0, 2.0) - -20.0).abs() < 1e-3);
    }

    /// The ring must read back oldest-left / newest-right: writing more columns than
    /// it holds and walking it in display order has to give the last `w` in order.
    #[test]
    fn ring_presents_columns_in_display_order() {
        let ctx = egui::Context::default();
        let (w, h) = (16usize, 4usize);
        let mut eng = Engine::new();
        eng.spectro_img = ColorImage::filled([w, h], BG);
        eng.spectro_size = [w, h];
        eng.spectro_tex = Some(ctx.load_texture(
            "t",
            ColorImage::filled([w, h], BG),
            TextureOptions::LINEAR,
        ));
        eng.rebuild_row_bands(48_000);

        // Each column is flat at a distinct level, so its color identifies it.
        let level = |i: usize| -100.0 + i as f32 * 2.0;
        let n = w + 5; // deliberately wrap past the end
        for i in 0..n {
            let col = vec![level(i); SPECTRUM_BINS];
            eng.push_spectro_columns(std::slice::from_ref(&col));
        }

        for j in 0..w {
            let ring = (eng.spectro_head + j) % w;
            let expected = n - w + j; // the last w columns written, in order
            let t = ((level(expected) - SPECTRO_FLOOR) / (SPECTRO_CEIL - SPECTRO_FLOOR))
                .clamp(0.0, 1.0);
            assert_eq!(
                eng.spectro_img.pixels[ring],
                magma(t),
                "display column {j} holds the wrong data"
            );
        }
    }
}
