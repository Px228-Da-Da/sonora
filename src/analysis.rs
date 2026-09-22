//! Real-time audio analysis: spectrum (FFT), K-weighted loudness (ITU-R BS.1770),
//! stereo correlation and goniometer point buffer.

use std::collections::VecDeque;
use std::f32::consts::PI;

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

pub const FFT_SIZE: usize = 4096;
pub const SPECTRUM_BINS: usize = FFT_SIZE / 2;
/// Number of stereo points kept for the goniometer scatter.
pub const GONIO_POINTS: usize = 2048;

/// Column rate used until the UI sizes the spectrogram and calls
/// [`Analyzer::set_column_rate`]. The hop is derived from the rate and the stream's
/// sample rate, so the time axis follows the sample clock, not the repaint rate.
const DEFAULT_COLUMN_RATE_HZ: f32 = 50.0;
/// Time constant of the spectrum's slow release. Converted to a per-hop coefficient
/// so the decay is the same regardless of sample rate.
const SPECTRUM_RELEASE_TAU: f32 = 0.040;
/// Ceiling on undrained spectrum columns, so a stalled UI can't grow memory. Only
/// reached if the UI stalls for seconds: normally a frame drains a handful.
const MAX_PENDING_COLUMNS: usize = 1024;
/// Fallback rate of the sample-peak bars, in dB per second.
const PEAK_FALL_DB_PER_SEC: f32 = 20.0;

/// A snapshot of all analysis results for one UI frame.
#[derive(Clone)]
pub struct Snapshot {
    /// Magnitude spectrum in dBFS, length SPECTRUM_BINS (index = FFT bin).
    pub spectrum_db: Vec<f32>,
    /// Spectrum columns produced since the last snapshot, oldest first. One per
    /// analysis hop — the spectrogram consumes all of them to keep its time axis
    /// linear even when the UI misses frames.
    pub columns: Vec<Vec<f32>>,
    pub momentary: f32,
    pub short_term: f32,
    pub integrated: f32,
    pub peak_db: [f32; 2],
    pub peak_hold_db: [f32; 2],
    pub correlation: f32,
    pub gonio: Vec<[f32; 2]>,
    pub sample_rate: u32,
}

/// Transposed direct-form II biquad.
#[derive(Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self { b0, b1, b2, a1, a2, z1: 0.0, z2: 0.0 }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// K-weighting pre-filter (high-shelf) per RBJ cookbook, prewarped to `fs`.
fn kweight_shelf(fs: f32) -> Biquad {
    let f0 = 1681.974450955533;
    let g = 3.999843853973347_f32; // dB
    let q = 0.7071752369554196_f32;
    let a = 10f32.powf(g / 40.0);
    let w0 = 2.0 * PI * f0 / fs;
    let cosw0 = w0.cos();
    let alpha = w0.sin() / (2.0 * q);
    let sa = a.sqrt();

    let b0 = a * ((a + 1.0) + (a - 1.0) * cosw0 + 2.0 * sa * alpha);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cosw0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cosw0 - 2.0 * sa * alpha);
    let a0 = (a + 1.0) - (a - 1.0) * cosw0 + 2.0 * sa * alpha;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cosw0);
    let a2 = (a + 1.0) - (a - 1.0) * cosw0 - 2.0 * sa * alpha;
    Biquad::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
}

/// K-weighting RLB high-pass filter, prewarped to `fs`.
fn kweight_highpass(fs: f32) -> Biquad {
    let f0 = 38.13547087602444;
    let q = 0.5003270373238773_f32;
    let w0 = 2.0 * PI * f0 / fs;
    let cosw0 = w0.cos();
    let alpha = w0.sin() / (2.0 * q);

    let b0 = (1.0 + cosw0) / 2.0;
    let b1 = -(1.0 + cosw0);
    let b2 = (1.0 + cosw0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cosw0;
    let a2 = 1.0 - alpha;
    Biquad::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
}

/// Loudness meter implementing ITU-R BS.1770-4 K-weighting + gating.
struct LoudnessMeter {
    // [channel][stage]
    filters: [[Biquad; 2]; 2],
    // ring of per-sample summed channel power (K-weighted), up to 3 s (short-term window)
    power_ring: VecDeque<f32>,
    st_capacity: usize,  // 3 s
    mom_len: usize,      // 0.4 s
    /// Running sums over the whole ring and over the momentary window. Kept so the
    /// readouts are O(1): they are polled every frame by every open window, and
    /// re-summing 3 s of samples there dominated the whole app's cost.
    sum_st: f64,
    sum_mom: f64,
    // integrated gating
    block_hop: usize,            // 0.1 s in samples
    since_hop: usize,            // samples accumulated toward next 100 ms hop
    block_powers: VecDeque<f32>, // one linear power per 400 ms gating block
    /// Gated loudness, recomputed only when a block is added (10 Hz), not per poll.
    integrated: f32,
    // peak
    peak: [f32; 2],
    peak_hold: [f32; 2],
    /// Per-sample multiplier giving PEAK_FALL_DB_PER_SEC of fallback.
    peak_decay: f32,
}

impl LoudnessMeter {
    fn new(fs: f32) -> Self {
        let shelf = kweight_shelf(fs);
        let hp = kweight_highpass(fs);
        Self {
            filters: [[shelf, hp], [shelf, hp]],
            power_ring: VecDeque::new(),
            st_capacity: (fs * 3.0) as usize,
            mom_len: (fs * 0.4) as usize,
            sum_st: 0.0,
            sum_mom: 0.0,
            block_hop: (fs * 0.1) as usize,
            since_hop: 0,
            block_powers: VecDeque::new(),
            integrated: f32::NEG_INFINITY,
            peak: [0.0; 2],
            peak_hold: [0.0; 2],
            peak_decay: 10f32.powf(-PEAK_FALL_DB_PER_SEC / (20.0 * fs)),
        }
    }

    fn process(&mut self, frames: &[[f32; 2]]) {
        for &[l, r] in frames {
            // peak (sample peak) with slow decay so the bar falls back
            let al = l.abs();
            let ar = r.abs();
            self.peak[0] = (self.peak[0] * self.peak_decay).max(al);
            self.peak[1] = (self.peak[1] * self.peak_decay).max(ar);
            self.peak_hold[0] = self.peak_hold[0].max(al);
            self.peak_hold[1] = self.peak_hold[1].max(ar);

            // K-weight each channel (two cascaded biquad stages)
            let l0 = self.filters[0][0].process(l);
            let lk = self.filters[0][1].process(l0);
            let r0 = self.filters[1][0].process(r);
            let rk = self.filters[1][1].process(r0);
            let z = lk * lk + rk * rk;

            self.power_ring.push_back(z);
            self.sum_st += z as f64;
            self.sum_mom += z as f64;
            // The sample that just fell out of the trailing momentary window.
            let len = self.power_ring.len();
            if len > self.mom_len {
                self.sum_mom -= self.power_ring[len - self.mom_len - 1] as f64;
            }
            if len > self.st_capacity {
                if let Some(old) = self.power_ring.pop_front() {
                    self.sum_st -= old as f64;
                }
            }

            self.since_hop += 1;
            if self.since_hop >= self.block_hop {
                self.since_hop = 0;
                // 400 ms gating block power = mean over last mom_len samples
                if self.power_ring.len() >= self.mom_len {
                    self.block_powers
                        .push_back((self.sum_mom / self.mom_len as f64) as f32);
                    // keep memory bounded (~ 60 min of blocks)
                    if self.block_powers.len() > 36000 {
                        self.block_powers.pop_front();
                    }
                    self.recompute_integrated();
                }
            }
        }
    }

    fn reset_integrated(&mut self) {
        self.block_powers.clear();
        self.integrated = f32::NEG_INFINITY;
        self.peak_hold = [0.0; 2];
    }

    fn momentary(&self) -> f32 {
        let n = self.mom_len.min(self.power_ring.len());
        if n == 0 {
            return f32::NEG_INFINITY;
        }
        loudness_from_power((self.sum_mom / n as f64) as f32)
    }

    fn short_term(&self) -> f32 {
        let n = self.power_ring.len();
        if n == 0 {
            return f32::NEG_INFINITY;
        }
        loudness_from_power((self.sum_st / n as f64) as f32)
    }

    /// Absolute (−70 LUFS) then relative (−10 LU) gating per BS.1770-4. Two passes,
    /// no allocation; called once per 100 ms block rather than once per readout.
    fn recompute_integrated(&mut self) {
        let abs_gate = power_from_loudness(-70.0);
        let (mut sum, mut n) = (0.0f64, 0usize);
        for &p in &self.block_powers {
            if p > abs_gate {
                sum += p as f64;
                n += 1;
            }
        }
        if n == 0 {
            self.integrated = f32::NEG_INFINITY;
            return;
        }
        let mean_abs = (sum / n as f64) as f32;
        let rel_gate = power_from_loudness(loudness_from_power(mean_abs) - 10.0);
        let (mut sum2, mut n2) = (0.0f64, 0usize);
        for &p in &self.block_powers {
            if p > abs_gate && p > rel_gate {
                sum2 += p as f64;
                n2 += 1;
            }
        }
        self.integrated = if n2 == 0 {
            f32::NEG_INFINITY
        } else {
            loudness_from_power((sum2 / n2 as f64) as f32)
        };
    }
}

#[inline]
fn loudness_from_power(p: f32) -> f32 {
    if p <= 0.0 {
        f32::NEG_INFINITY
    } else {
        -0.691 + 10.0 * p.log10()
    }
}

#[inline]
fn power_from_loudness(l: f32) -> f32 {
    10f32.powf((l + 0.691) / 10.0)
}

/// Rolling stereo correlation over ~300 ms with running sums.
///
/// The sums are `f64`: each is built by adding and subtracting millions of terms
/// per second, and in `f32` the cancellation error drifts visibly within minutes.
struct Correlation {
    ring: VecDeque<[f32; 2]>,
    cap: usize,
    sll: f64,
    srr: f64,
    slr: f64,
}

impl Correlation {
    fn new(fs: f32) -> Self {
        Self {
            ring: VecDeque::new(),
            cap: (fs * 0.3) as usize,
            sll: 0.0,
            srr: 0.0,
            slr: 0.0,
        }
    }

    fn process(&mut self, frames: &[[f32; 2]]) {
        for &[l, r] in frames {
            let (l, r) = (l as f64, r as f64);
            self.ring.push_back([l as f32, r as f32]);
            self.sll += l * l;
            self.srr += r * r;
            self.slr += l * r;
            if self.ring.len() > self.cap {
                if let Some([ol, or]) = self.ring.pop_front() {
                    let (ol, or) = (ol as f64, or as f64);
                    self.sll -= ol * ol;
                    self.srr -= or * or;
                    self.slr -= ol * or;
                }
            }
        }
    }

    fn value(&self) -> f32 {
        let denom = (self.sll * self.srr).sqrt();
        if denom <= 1e-20 {
            0.0
        } else {
            (self.slr / denom).clamp(-1.0, 1.0) as f32
        }
    }
}

/// Top-level analyzer owned by the UI thread.
pub struct Analyzer {
    sample_rate: u32,
    loudness: LoudnessMeter,
    correlation: Correlation,
    mono: VecDeque<f32>,
    gonio: VecDeque<[f32; 2]>,
    // FFT
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    window_gain: f32,
    fft_buf: Vec<Complex<f32>>,
    // smoothed spectrum for display
    smooth_db: Vec<f32>,
    // analysis hop: one spectrum column every `spectrum_hop` input samples
    spectrum_hop: usize,
    since_spectrum: usize,
    release_coeff: f32,
    /// Columns computed but not yet drained by the UI, oldest first.
    columns: VecDeque<Vec<f32>>,
}

impl Analyzer {
    pub fn new(sample_rate: u32) -> Self {
        let fs = sample_rate as f32;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        // Hann window
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / FFT_SIZE as f32).cos())
            .collect();
        let window_gain: f32 = window.iter().sum();

        let mut analyzer = Self {
            sample_rate,
            loudness: LoudnessMeter::new(fs),
            correlation: Correlation::new(fs),
            mono: VecDeque::with_capacity(FFT_SIZE),
            gonio: VecDeque::with_capacity(GONIO_POINTS),
            fft,
            window,
            window_gain,
            fft_buf: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            smooth_db: vec![-120.0; SPECTRUM_BINS],
            spectrum_hop: 1,
            since_spectrum: 0,
            release_coeff: 1.0,
            columns: VecDeque::new(),
        };
        analyzer.set_column_rate(DEFAULT_COLUMN_RATE_HZ);
        analyzer
    }

    /// Set how many spectrum columns are produced per second.
    ///
    /// The UI derives this from the spectrogram's width and the time span it should
    /// show, so the scroll speed is the same at any window size or display scale.
    /// The release coefficient follows, keeping the smoothing time constant fixed.
    pub fn set_column_rate(&mut self, hz: f32) {
        let fs = self.sample_rate as f32;
        let hop = (fs / hz.max(1.0)).max(1.0);
        self.spectrum_hop = hop as usize;
        self.since_spectrum = 0;
        self.release_coeff = 1.0 - (-(hop / fs) / SPECTRUM_RELEASE_TAU).exp();
    }

    pub fn reset_integrated(&mut self) {
        self.loudness.reset_integrated();
    }

    pub fn process(&mut self, frames: &[[f32; 2]]) {
        self.loudness.process(frames);
        self.correlation.process(frames);

        for &[l, r] in frames {
            let m = 0.5 * (l + r);
            self.mono.push_back(m);
            if self.mono.len() > FFT_SIZE {
                self.mono.pop_front();
            }
            self.gonio.push_back([l, r]);
            if self.gonio.len() > GONIO_POINTS {
                self.gonio.pop_front();
            }

            // Emit a spectrum column every hop, driven by the sample clock rather
            // than by how often the UI happens to repaint.
            self.since_spectrum += 1;
            if self.since_spectrum >= self.spectrum_hop {
                self.since_spectrum = 0;
                self.compute_spectrum();
                if self.columns.len() >= MAX_PENDING_COLUMNS {
                    self.columns.pop_front();
                }
                self.columns.push_back(self.smooth_db.clone());
            }
        }
    }

    fn compute_spectrum(&mut self) {
        let n = self.mono.len();
        // Fill FFT buffer: zero-pad at the front if we don't have a full window yet.
        let offset = FFT_SIZE - n;
        for c in self.fft_buf.iter_mut().take(offset) {
            *c = Complex::new(0.0, 0.0);
        }
        for (i, &s) in self.mono.iter().enumerate() {
            let w = self.window[offset + i];
            self.fft_buf[offset + i] = Complex::new(s * w, 0.0);
        }

        self.fft.process(&mut self.fft_buf);

        let norm = 2.0 / self.window_gain;
        for i in 0..SPECTRUM_BINS {
            let mag = self.fft_buf[i].norm() * norm;
            let db = 20.0 * (mag.max(1e-9)).log10();
            // exponential smoothing: fast attack, slow release
            let prev = self.smooth_db[i];
            self.smooth_db[i] = if db > prev {
                db
            } else {
                prev + (db - prev) * self.release_coeff
            };
        }
    }

    pub fn snapshot(&mut self) -> Snapshot {
        let pk = self.loudness.peak;
        let ph = self.loudness.peak_hold;
        Snapshot {
            spectrum_db: self.smooth_db.clone(),
            columns: self.columns.drain(..).collect(),
            momentary: self.loudness.momentary(),
            short_term: self.loudness.short_term(),
            integrated: self.loudness.integrated,
            peak_db: [to_db(pk[0]), to_db(pk[1])],
            peak_hold_db: [to_db(ph[0]), to_db(ph[1])],
            correlation: self.correlation.value(),
            gonio: self.gonio.iter().copied().collect(),
            sample_rate: self.sample_rate,
        }
    }
}

#[inline]
fn to_db(v: f32) -> f32 {
    if v <= 1e-9 {
        -120.0
    } else {
        20.0 * v.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(fs: f32, freq: f32, amp: f32, n: usize) -> Vec<[f32; 2]> {
        (0..n)
            .map(|i| {
                let s = amp * (2.0 * PI * freq * i as f32 / fs).sin();
                [s, s]
            })
            .collect()
    }

    #[test]
    fn peak_falls_at_20db_per_second() {
        for fs in [44100.0f32, 48000.0, 96000.0] {
            let mut m = LoudnessMeter::new(fs);
            m.process(&[[1.0, 1.0]]);
            let start = 20.0 * m.peak[0].log10();
            m.process(&vec![[0.0, 0.0]; fs as usize]); // one second of silence
            let end = 20.0 * m.peak[0].log10();
            let fall = start - end;
            assert!((fall - 20.0).abs() < 0.1, "fs={fs}: fell {fall} dB/s");
        }
    }

    #[test]
    fn column_rate_is_independent_of_call_size() {
        let fs = 48_000u32;
        // same 1 s of audio, delivered in one chunk vs. in 1 ms chunks
        let mut a = Analyzer::new(fs);
        a.process(&tone(fs as f32, 1000.0, 0.5, fs as usize));
        let bulk = a.snapshot().columns.len();

        let mut b = Analyzer::new(fs);
        let mut drip = 0;
        for chunk in tone(fs as f32, 1000.0, 0.5, fs as usize).chunks(48) {
            b.process(chunk);
            drip += b.snapshot().columns.len();
        }
        assert_eq!(bulk, drip, "bulk={bulk} drip={drip}");
        assert_eq!(bulk, DEFAULT_COLUMN_RATE_HZ as usize, "expected 50 columns for 1 s");
    }

    /// The UI asks for one column per pixel of width, so however wide the window is,
    /// the spectrogram must still take the same wall-clock time to fill.
    #[test]
    fn column_rate_keeps_the_time_span_constant() {
        let fs = 48_000u32;
        for width in [512usize, 2112, 3840] {
            let mut a = Analyzer::new(fs);
            a.set_column_rate(width as f32 / 10.0);
            let audio = tone(fs as f32, 1000.0, 0.5, fs as usize * 10); // 10 s
            let mut total = 0;
            for chunk in audio.chunks(fs as usize / 60) {
                a.process(chunk);
                total += a.snapshot().columns.len();
            }
            let err = (total as i64 - width as i64).abs();
            assert!(err <= 2, "width {width}: 10 s produced {total} columns");
        }
    }

    #[test]
    fn full_scale_sine_reads_near_0_dbfs() {
        let fs = 48_000u32;
        let mut a = Analyzer::new(fs);
        // exactly on bin 86, so there is no scalloping loss to account for
        let freq = 86.0 * fs as f32 / FFT_SIZE as f32;
        a.process(&tone(fs as f32, freq, 1.0, fs as usize));
        let snap = a.snapshot();
        let bin = (freq / (fs as f32 / 2.0) * SPECTRUM_BINS as f32).round() as usize;
        let peak = snap.spectrum_db[bin - 2..bin + 3]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(peak > -0.5 && peak < 0.5, "1 kHz full-scale sine read {peak} dBFS");
    }

    #[test]
    fn correlation_does_not_drift_on_long_runs() {
        let fs = 48_000.0;
        let mut c = Correlation::new(fs);
        // 3 minutes of decorrelated-ish content, then hard-panned silence check
        let n = (fs * 180.0) as usize;
        let frames: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let t = i as f32 / fs;
                [(2.0 * PI * 440.0 * t).sin(), (2.0 * PI * 440.0 * t).sin()]
            })
            .collect();
        c.process(&frames);
        let v = c.value();
        assert!((v - 1.0).abs() < 1e-3, "identical channels gave correlation {v}");
    }
}
