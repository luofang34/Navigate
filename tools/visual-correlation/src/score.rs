//! Masked mean dot scores. Positive lag samples the map to the right and below.
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid grid dimensions, mask, or values")]
    Grid,
    #[error("incompatible query or transform dimensions")]
    Shape,
}
#[derive(Clone)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub c: usize,
    pub values: Vec<f32>,
    pub mask: Vec<bool>,
}
impl Grid {
    pub fn new(
        w: usize,
        h: usize,
        c: usize,
        mut values: Vec<f32>,
        mask: Vec<bool>,
    ) -> Result<Self, Error> {
        if w == 0
            || h == 0
            || c == 0
            || w > 256
            || h > 256
            || c > 64
            || values.len() != w * h * c
            || mask.len() != w * h
        {
            return Err(Error::Grid);
        }
        for (cell, valid) in mask.iter().enumerate() {
            for v in &mut values[cell * c..(cell + 1) * c] {
                if !valid {
                    *v = 0.0;
                } else if !v.is_finite() {
                    return Err(Error::Grid);
                }
            }
        }
        Ok(Self {
            w,
            h,
            c,
            values,
            mask,
        })
    }
    pub fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Result<Self, Error> {
        if x + w > self.w || y + h > self.h {
            return Err(Error::Grid);
        }
        let mut values = Vec::new();
        let mut mask = Vec::new();
        for row in y..y + h {
            for col in x..x + w {
                let i = row * self.w + col;
                values.extend_from_slice(&self.values[i * self.c..(i + 1) * self.c]);
                mask.push(self.mask[i]);
            }
        }
        Self::new(w, h, self.c, values, mask)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Score {
    pub count: usize,
    pub value: Option<f64>,
}

pub fn direct(q: &Grid, m: &Grid, lags: &[[i32; 2]], min: usize, f64_acc: bool) -> Vec<Score> {
    lags.iter()
        .map(|&[dx, dy]| {
            let mut count = 0usize;
            let mut sum64 = 0.0f64;
            let mut sum32 = [0.0f32; 4];
            for y in 0..q.h {
                for x in 0..q.w {
                    let mx = x as i32 + dx;
                    let my = y as i32 + dy;
                    if mx < 0 || my < 0 || mx >= m.w as i32 || my >= m.h as i32 {
                        continue;
                    }
                    let qi = y * q.w + x;
                    let mi = my as usize * m.w + mx as usize;
                    if !q.mask[qi] || !m.mask[mi] {
                        continue;
                    }
                    count = count.wrapping_add(1);
                    let a = &q.values[qi * q.c..(qi + 1) * q.c];
                    let b = &m.values[mi * m.c..(mi + 1) * m.c];
                    if f64_acc {
                        for (a, b) in a.iter().zip(b) {
                            sum64 += f64::from(*a) * f64::from(*b);
                        }
                    } else {
                        let (ac, ar) = a.as_chunks::<4>();
                        let (bc, br) = b.as_chunks::<4>();
                        for (a, b) in ac.iter().zip(bc) {
                            for lane in 0..4 {
                                sum32[lane] += a[lane] * b[lane];
                            }
                        }
                        for (a, b) in ar.iter().zip(br) {
                            sum32[0] += a * b;
                        }
                    }
                }
            }
            Score {
                count,
                value: (count >= min && count > 0).then(|| {
                    if f64_acc {
                        sum64 / count as f64
                    } else {
                        f64::from(sum32.iter().sum::<f32>() / count as f32)
                    }
                }),
            }
        })
        .collect()
}

struct Plan {
    w: usize,
    h: usize,
    fw: Arc<dyn Fft<f32>>,
    fh: Arc<dyn Fft<f32>>,
    iw: Arc<dyn Fft<f32>>,
    ih: Arc<dyn Fft<f32>>,
    scratch: Vec<Complex<f32>>,
    column: Vec<Complex<f32>>,
}
impl Plan {
    fn new(w: usize, h: usize) -> Self {
        let mut p = FftPlanner::new();
        let fw = p.plan_fft_forward(w);
        let fh = p.plan_fft_forward(h);
        let iw = p.plan_fft_inverse(w);
        let ih = p.plan_fft_inverse(h);
        let n = [&fw, &fh, &iw, &ih]
            .iter()
            .map(|p| p.get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        Self {
            w,
            h,
            fw,
            fh,
            iw,
            ih,
            scratch: vec![Complex::default(); n],
            column: vec![Complex::default(); h],
        }
    }
    fn apply(&mut self, data: &mut [Complex<f32>], inverse: bool) {
        let (row, col) = if inverse {
            (&self.iw, &self.ih)
        } else {
            (&self.fw, &self.fh)
        };
        for line in data.chunks_exact_mut(self.w) {
            row.process_with_scratch(line, &mut self.scratch);
        }
        for x in 0..self.w {
            for y in 0..self.h {
                self.column[y] = data[y * self.w + x];
            }
            col.process_with_scratch(&mut self.column, &mut self.scratch);
            for y in 0..self.h {
                data[y * self.w + x] = self.column[y];
            }
        }
    }
    fn channel(&mut self, g: &Grid, c: usize) -> Vec<Complex<f32>> {
        let mut a = vec![Complex::default(); self.w * self.h];
        for y in 0..g.h {
            for x in 0..g.w {
                let i = y * g.w + x;
                a[y * self.w + x].re = if c == g.c {
                    if g.mask[i] { 1.0 } else { 0.0 }
                } else {
                    g.values[i * g.c + c]
                };
            }
        }
        self.apply(&mut a, false);
        a
    }
}

pub struct Correlator {
    plan: Plan,
    map: Vec<Vec<Complex<f32>>>,
    qw: usize,
    qh: usize,
    mw: usize,
    mh: usize,
    c: usize,
}
impl Correlator {
    pub fn new(q: &Grid, m: &Grid) -> Result<Self, Error> {
        if q.c != m.c {
            return Err(Error::Shape);
        }
        let mut plan = Plan::new(
            (q.w + m.w - 1).next_power_of_two(),
            (q.h + m.h - 1).next_power_of_two(),
        );
        let map = (0..=m.c).map(|c| plan.channel(m, c)).collect();
        Ok(Self {
            plan,
            map,
            qw: q.w,
            qh: q.h,
            mw: m.w,
            mh: m.h,
            c: q.c,
        })
    }
    pub fn cached_bytes(&self) -> usize {
        self.map.iter().map(|v| v.capacity() * 8).sum()
    }
    pub fn score(&mut self, q: &Grid, lags: &[[i32; 2]], min: usize) -> Result<Vec<Score>, Error> {
        if q.w != self.qw || q.h != self.qh || q.c != self.c {
            return Err(Error::Shape);
        }
        let n = self.plan.w * self.plan.h;
        let mut numerator = vec![Complex::default(); n];
        for c in 0..q.c {
            let f = self.plan.channel(q, c);
            for (i, v) in f.iter().enumerate() {
                numerator[i] += v.conj() * self.map[c][i];
            }
        }
        self.plan.apply(&mut numerator, true);
        let mut counts = self.plan.channel(q, q.c);
        for (i, v) in counts.iter_mut().enumerate() {
            *v = v.conj() * self.map[q.c][i];
        }
        self.plan.apply(&mut counts, true);
        Ok(lags
            .iter()
            .map(|&[x, y]| {
                if x <= -(q.w as i32)
                    || y <= -(q.h as i32)
                    || x >= self.mw as i32
                    || y >= self.mh as i32
                {
                    return Score {
                        count: 0,
                        value: None,
                    };
                }
                let i = y.rem_euclid(self.plan.h as i32) as usize * self.plan.w
                    + x.rem_euclid(self.plan.w as i32) as usize;
                let count = (counts[i].re / n as f32).round().max(0.0) as usize;
                Score {
                    count,
                    value: (count >= min && count > 0)
                        .then(|| f64::from(numerator[i].re / n as f32 / count as f32)),
                }
            })
            .collect())
    }
}
#[cfg(test)]
#[path = "score/tests.rs"]
mod tests;
