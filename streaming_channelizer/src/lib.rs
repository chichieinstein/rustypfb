use bessel_fun_sys::{bessel_func};
use num::{Complex, Zero};
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;
use std::time::Instant;

fn channel_fn(ind: usize, nchannel: usize, nproto: usize, kbeta: f32) -> f32 {
    let ind_arg = ind as f32;
    let arg = -((nproto / 2) as f32) + (ind_arg + 1.0) / (nchannel as f32);
    let darg = (2.0 * ind_arg) / ((nchannel * nproto) as f32) - 1.0;
    let carg = kbeta * (1.0 - darg * darg).sqrt();
    (unsafe { bessel_func(carg) }) / (unsafe { bessel_func(kbeta) })
        * (if arg != 0.0 { arg.sin() / arg } else { 1.0 })
}

// fn create_filter<const QUADRUPLE_TAPS: usize>(channels: usize) -> [f32; QUADRUPLE_TAPS] {
//     let mut result = [f32::zero(); QUADRUPLE_TAPS];
//     let taps = QUADRUPLE_TAPS / (4 * channels);
//     for chann_id in 0..channels {
//         for tap_id in 0..taps {
//             let ind = tap_id * channels + chann_id;
//             if chann_id < channels / 2 {
//                 result[4 * taps * chann_id + 4 * tap_id] = channel_fn(ind, channels, taps, 10.0);
//                 result[4 * taps * chann_id + 4 * tap_id + 1] =
//                     channel_fn(ind, channels, taps, 10.0);
//             } else {
//                 result[4 * taps * chann_id + 4 * tap_id + 2] =
//                     channel_fn(ind, channels, taps, 10.0);
//                 result[4 * taps * chann_id + 4 * tap_id + 3] =
//                     channel_fn(ind, channels, taps, 10.0);
//             }
//         }
//     }
//     result
// }

fn create_filter_chunk<const TWICE_TAPS: usize>(
    channels: usize,
    slice: usize,
) -> Vec<Complex<f32>> {
    let mut result = vec![Complex::<f32>::zero(); channels * slice];
    let taps = TWICE_TAPS / 2;
    let mut fft_planner = FftPlanner::new();
    let mut fft = fft_planner.plan_fft_forward(slice);
    for chann_id in 0..channels {
        let buffer = &mut result[chann_id * slice..(chann_id + 1) * slice];
        for tap_id in 0..taps {
            let ind = tap_id * channels + chann_id;
            if chann_id < channels / 2 {
                buffer[2 * tap_id] =
                    Complex::<f32>::new(channel_fn(ind, channels, taps, 10.0), 0.0);
            } else {
                buffer[2 * tap_id + 1] =
                    Complex::<f32>::new(channel_fn(ind, channels, taps, 10.0), 0.0);
            }
        }
        fft.process(buffer);
    }
    result
}

#[derive(Copy, Clone, Debug)]
pub struct ChannState<T, const CHUNK_SIZE: usize, const HOP_SIZE: usize> {
    buffer: [T; CHUNK_SIZE],
}

impl<T: Default + Copy, const CHUNK_SIZE: usize, const HOP_SIZE: usize>
    ChannState<T, CHUNK_SIZE, HOP_SIZE>
{
    fn new() -> Self {
        Self {
            buffer: [T::default(); CHUNK_SIZE],
            // flag: true,
        }
    }

    #[inline]
    fn add(&mut self, samples: &[T]) {
        self.buffer.rotate_left(HOP_SIZE);
        self.buffer[CHUNK_SIZE - HOP_SIZE..].clone_from_slice(&samples[..HOP_SIZE]);
    }
}

pub struct StreamingChannelizer<
    const CHUNK_SIZE: usize,
    const TWICE_TAPS: usize,
    const HOP_SIZE: usize,
> {
    state: ChannState<Complex<f32>, CHUNK_SIZE, HOP_SIZE>,
    inp_scratch: [Complex<f32>; CHUNK_SIZE],
    filter_scratch: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    downconvert_fft_scratch: Vec<Complex<f32>>,
    channels: i32,
    filter: Vec<Complex<f32>>,
    output: Vec<Complex<f32>>,
    output_scratch: Vec<Complex<f32>>,
    fft_initial: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    fft_downconvert: Arc<dyn Fft<f32>>,
}

impl<const CHUNK_SIZE: usize, const TWICE_TAPS: usize, const HOP_SIZE: usize>
    StreamingChannelizer<CHUNK_SIZE, TWICE_TAPS, HOP_SIZE>
{
    pub fn new(channels: i32) -> Self {
        let state = ChannState::<Complex<f32>, CHUNK_SIZE, HOP_SIZE>::new();
        let scratch = [Complex::<f32>::zero(); CHUNK_SIZE];
        let filter_scratch = vec![Complex::<f32>::zero(); 2 * CHUNK_SIZE];
        let fft_scratch = vec![Complex::<f32>::zero(); (2 * CHUNK_SIZE) / (channels as usize)];
        let downconvert_fft_scratch = vec![Complex::<f32>::zero(); channels as usize];
        let output_scratch = vec![Complex::<f32>::zero(); channels as usize];
        let mut output = vec![Complex::<f32>::zero(); 2 * CHUNK_SIZE];
        let mut plan = FftPlanner::new();

        StreamingChannelizer {
            state,
            inp_scratch: scratch,
            filter_scratch,
            filter: create_filter_chunk::<TWICE_TAPS>(
                channels as usize,
                2 * CHUNK_SIZE / (channels as usize),
            ),
            channels,
            fft_scratch,
            downconvert_fft_scratch,
            fft_initial: plan.plan_fft_forward(2 * CHUNK_SIZE / (channels as usize)),
            fft_inverse: plan.plan_fft_inverse(CHUNK_SIZE / (channels as usize)),
            fft_downconvert: plan.plan_fft_inverse(channels as usize),
            output_scratch,
            output,
        }
    }

    pub fn add(&mut self, samples: &[Complex<f32>]) {
        self.state.add(samples);
    }

    pub fn process(&mut self) {
        let nslice = 2 * (CHUNK_SIZE) / (self.channels as usize);
        self.inp_scratch
            .chunks_mut(nslice)
            .zip((0..self.channels / 2).map(|ind| {
                self.state.buffer[ind as usize..]
                    .iter()
                    .step_by((self.channels / 2) as usize)
            }))
            .for_each(|(x, y)| {
                y.into_iter().zip(x).for_each(|(yl, xl)| *xl = *yl);
            });
        self.inp_scratch
            .chunks_mut((2 * CHUNK_SIZE / (self.channels as usize)))
            .for_each(|chunk| {
                self.fft_initial
                    .process_with_scratch(chunk, &mut self.fft_scratch)
            });
        self.filter
            .iter()
            .zip(self.inp_scratch.iter().chain(self.inp_scratch.iter()))
            .zip(self.filter_scratch.iter_mut())
            .for_each(|((filter, input), output)| *output = input * filter);

        self.filter_scratch
            .chunks_mut((2 * CHUNK_SIZE / (self.channels as usize)))
            .for_each(|chunk| {
                self.fft_inverse
                    .process_with_scratch(chunk, &mut self.fft_scratch)
            });
        (0..nslice).for_each(|ind| {
            self.filter_scratch[ind..]
                .iter()
                .step_by(nslice as usize)
                .map(|x| x.clone())
                .zip(self.output_scratch.iter_mut())
                .for_each(|(y, z)| *z = y);
            self.fft_downconvert
                .process_with_scratch(&mut self.output_scratch, &mut self.downconvert_fft_scratch);
            self.output[ind..]
                .iter_mut()
                .step_by(nslice as usize)
                .zip(self.output_scratch.iter())
                .for_each(|(o_el, w_elem)| *o_el = *w_elem);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::Zero;

    const CHANNELS: usize = 1024;
    const CHUNK_SIZE: usize = 8192*2;
    const HOP_SIZE: usize = 512;
    const TWICE_TAPS: usize = 32;

    #[test]
    fn state_timing() {
        // let howmany = 1;
        let NTIMES = 100 as usize;
        let mut channelizer =
            StreamingChannelizer::<CHUNK_SIZE, TWICE_TAPS, HOP_SIZE>::new(CHANNELS as i32);

        // let mut samples = vec![Complex::<f32>::zero(); CHUNK_SIZE];
        let mut others = vec![Complex::<f32>::zero(); CHUNK_SIZE];

        let now = Instant::now();
        for _ in 0..NTIMES {
            channelizer.add(&others);
            channelizer.process();
        }
        println!("{:?}", now.elapsed().as_secs_f32() / (NTIMES as f32));
    }
}
