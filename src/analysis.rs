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

/// A snapshot of all analysis results for one UI frame.
#[derive(Clone)]
pub struct Snapshot {
    /// Magnitude spectrum in dBFS, length SPECTRUM_BINS (index = FFT bin).
    pub spectrum_db: Vec<f32>,
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
    // integrated gating
    block_hop: usize,        // 0.1 s in samples
    since_hop: usize,        // samples accumulated toward next 100 ms hop
    block_powers: Vec<f32>,  // one linear power per 400 ms gating block
    // peak
    peak: [f32; 2],
    peak_hold: [f32; 2],
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
            block_hop: (fs * 0.1) as usize,
            since_hop: 0,
            block_powers: Vec::new(),
            peak: [0.0; 2],
            peak_hold: [0.0; 2],
        }
    }

    fn process(&mut self, frames: &[[f32; 2]]) {
        for &[l, r] in frames {
            // peak (sample peak) with slow decay so the bar falls back
            let al = l.abs();
            let ar = r.abs();
            self.peak[0] = (self.peak[0] * 0.9995).max(al);
            self.peak[1] = (self.peak[1] * 0.9995).max(ar);
            self.peak_hold[0] = self.peak_hold[0].max(al);
            self.peak_hold[1] = self.peak_hold[1].max(ar);

            // K-weight each channel (two cascaded biquad stages)
            let l0 = self.filters[0][0].process(l);
            let lk = self.filters[0][1].process(l0);
            let r0 = self.filters[1][0].process(r);
            let rk = self.filters[1][1].process(r0);
            let z = lk * lk + rk * rk;

            self.power_ring.push_back(z);
            if self.power_ring.len() > self.st_capacity {
                self.power_ring.pop_front();
            }

            self.since_hop += 1;
            if self.since_hop >= self.block_hop {
                self.since_hop = 0;
                // 400 ms gating block power = mean over last mom_len samples
                let n = self.mom_len.min(self.power_ring.len());
                if n >= self.mom_len {
                    let sum: f32 = self.power_ring.iter().rev().take(n).sum();
                    self.block_powers.push(sum / n as f32);
                    // keep memory bounded (~ 60 min of blocks)
                    if self.block_powers.len() > 36000 {
                        self.block_powers.remove(0);
                    }
                }
            }
        }
    }

    fn reset_integrated(&mut self) {
        self.block_powers.clear();
        self.peak_hold = [0.0; 2];
    }

    fn momentary(&self) -> f32 {
        let n = self.mom_len.min(self.power_ring.len());
        if n == 0 {
            return f32::NEG_INFINITY;
        }
        let sum: f32 = self.power_ring.iter().rev().take(n).sum();
        loudness_from_power(sum / n as f32)
    }

    fn short_term(&self) -> f32 {
        let n = self.power_ring.len();
        if n == 0 {
            return f32::NEG_INFINITY;
        }
        let sum: f32 = self.power_ring.iter().sum();
        loudness_from_power(sum / n as f32)
    }

    fn integrated(&self) -> f32 {
        if self.block_powers.is_empty() {
            return f32::NEG_INFINITY;
        }
        // Absolute gate at -70 LUFS
        let abs_gate_power = power_from_loudness(-70.0);
        let above_abs: Vec<f32> = self
            .block_powers
            .iter()
            .copied()
            .filter(|&p| p > abs_gate_power)
            .collect();
        if above_abs.is_empty() {
            return f32::NEG_INFINITY;
        }
        let mean_abs: f32 = above_abs.iter().sum::<f32>() / above_abs.len() as f32;
        // Relative gate: mean loudness of abs-gated blocks minus 10 LU
        let rel_gate_power = power_from_loudness(loudness_from_power(mean_abs) - 10.0);
        let gated: Vec<f32> = above_abs.into_iter().filter(|&p| p > rel_gate_power).collect();
        if gated.is_empty() {
            return f32::NEG_INFINITY;
        }
        let mean: f32 = gated.iter().sum::<f32>() / gated.len() as f32;
        loudness_from_power(mean)
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
struct Correlation {
    ring: VecDeque<[f32; 2]>,
    cap: usize,
    sll: f32,
    srr: f32,
    slr: f32,
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
            self.ring.push_back([l, r]);
            self.sll += l * l;
            self.srr += r * r;
            self.slr += l * r;
            if self.ring.len() > self.cap {
                if let Some([ol, or]) = self.ring.pop_front() {
                    self.sll -= ol * ol;
                    self.srr -= or * or;
                    self.slr -= ol * or;
                }
            }
        }
    }

    fn value(&self) -> f32 {
        let denom = (self.sll * self.srr).sqrt();
        if denom <= 1e-12 {
            0.0
        } else {
            (self.slr / denom).clamp(-1.0, 1.0)
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

        Self {
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
        }
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
                prev + (db - prev) * 0.35
            };
        }
    }

    pub fn snapshot(&mut self) -> Snapshot {
        self.compute_spectrum();
        let pk = self.loudness.peak;
        let ph = self.loudness.peak_hold;
        Snapshot {
            spectrum_db: self.smooth_db.clone(),
            momentary: self.loudness.momentary(),
            short_term: self.loudness.short_term(),
            integrated: self.loudness.integrated(),
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
