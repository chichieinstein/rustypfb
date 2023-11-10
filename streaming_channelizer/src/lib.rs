use bessel_fun_sys::bessel_func;

use fftw_sys::{fftwf_execute, fftwf_plan, fftwf_plan_many_dft, FFTW_ESTIMATE};
use num::{Complex, Zero};
use rustfft::{Fft, FftPlanner};

use std::iter::Chain;
use std::slice::Iter;
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

fn create_filter_chunk<const TWICE_TAPS: usize, const CHUNK_SIZE: usize>(
    channels: usize,
) -> Vec<Complex<f32>> {
    let mut planner = FftPlanner::new();
    let mut plan = planner.plan_fft_forward(CHUNK_SIZE);
    let mut coeff = vec![Complex::new(0.0 as f32, 0.0); CHUNK_SIZE * channels];
    let taps = TWICE_TAPS / 2;
    for chann_id in 0..channels {
        // let buffer = &mut result[chann_id];
        let mut result = [Complex::<f32>::zero(); CHUNK_SIZE];
        for tap_id in 0..taps {
            let ind = tap_id * channels + chann_id;
            if chann_id < channels / 2 {
                result[2 * tap_id] = Complex::new(channel_fn(ind, channels, taps, 10.0), 0.0);
            } else {
                result[2 * tap_id + 1] = Complex::new(channel_fn(ind, channels, taps, 10.0), 0.0);
            }
        }
        plan.process(&mut result);
        coeff[chann_id * CHUNK_SIZE..(chann_id+1)*CHUNK_SIZE].clone_from_slice(&result);
    }
    coeff
}

fn create_filter<const TWICE_TAPS: usize>(channels: usize) -> Vec<[f32; TWICE_TAPS]> {
    let mut result = vec![[f32::zero(); TWICE_TAPS]; channels];
    let taps = TWICE_TAPS / 2;
    for chann_id in 0..channels {
        let buffer = &mut result[chann_id];
        for tap_id in 0..taps {
            let ind = tap_id * channels + chann_id;
            if chann_id < channels / 2 {
                buffer[2 * tap_id] = channel_fn(ind, channels, taps, 10.0);
            } else {
                buffer[2 * tap_id + 1] = channel_fn(ind, channels, taps, 10.0);
            }
        }
    }
    result
}

#[derive(Copy, Clone, Debug)]
struct Ring<T, const CAPACITY: usize> {
    head: usize,
    full: bool,
    buffer: [T; CAPACITY],
}

impl<T: Default + Copy, const CAPACITY: usize> Ring<T, CAPACITY> {
    fn new() -> Self {
        Self {
            head: 0,
            full: false,
            buffer: [T::default(); CAPACITY],
        }
    }

    #[inline]
    fn add(&mut self, element: T) {
        self.buffer[self.head] = element;
        self.head += 1;
        if self.head >= CAPACITY {
            self.head = 0;
            self.full = true;
        }
    }

    fn inner_iter(&self) -> Chain<Iter<'_, T>, Iter<'_, T>> {
        let initial = self.buffer[..self.head].iter();
        if self.full {
            return initial.chain(self.buffer[self.head..].iter());
        }
        initial.chain(self.buffer[0..0].iter())
    }

    #[inline]
    fn reset(&mut self) {
        self.head = 0;
        self.full = false;
    }
}

#[derive(Clone)]
pub struct Channelizer<const TWICE_TAPS: usize, const CHUNK_SIZE: usize> {
    channels: usize,
    fft: Arc<dyn Fft<f32>>,
    coeff: Vec<[f32; TWICE_TAPS]>,
    conv_coeff: Vec<Complex<f32>>,
    state: Vec<Ring<Complex<f32>, CHUNK_SIZE>>,
    scratch: Vec<Complex<f32>>,

    chunk_fft_input: Vec<Complex<f32>>,
    chunk_fft_output: Vec<Complex<f32>>,
    filter_fft_input: Vec<Complex<f32>>,
    chunk_output: Vec<Complex<f32>>,
}

pub enum FftList {
    Input,
    Filter,
    Downconvert,
}

pub fn plan_helper(
    input: &mut Vec<Complex<f32>>,
    output: &mut Vec<Complex<f32>>,
    nchannel: &i32,
    chunks: &i32,
    plan: FftList,
) -> fftwf_plan {
    match plan {
        FftList::Input => unsafe {
            fftwf_plan_many_dft(
                1,
                chunks,
                nchannel / 2,
                &mut input[0],
                chunks,
                1,
                *chunks,
                &mut output[0],
                chunks,
                1,
                *chunks,
                -1,
                FFTW_ESTIMATE,
            )
        },
        FftList::Filter => unsafe {
            fftwf_plan_many_dft(
                1,
                chunks,
                *nchannel,
                &mut input[0],
                chunks,
                1,
                *chunks,
                &mut output[0],
                chunks,
                1,
                *chunks,
                1,
                FFTW_ESTIMATE,
            )
        },
        FftList::Downconvert => unsafe {
            fftwf_plan_many_dft(
                1,
                nchannel,
                *chunks,
                &mut input[0],
                nchannel,
                *chunks,
                1,
                &mut output[0],
                nchannel,
                *chunks,
                *chunks,
                1,
                FFTW_ESTIMATE,
            )
        },
    }
}

pub struct ChannelizationPlans<const CHUNK_SIZE: usize> {
    forward_plan: fftwf_plan,
    reverse_plan: fftwf_plan,
    down_convert_plan: fftwf_plan,
    channels: i32,
}

impl<const CHUNK_SIZE: usize> ChannelizationPlans<CHUNK_SIZE> {
    pub fn new(
        chunk_fft_input: &mut Vec<Complex<f32>>,
        chunk_fft_output: &mut Vec<Complex<f32>>,
        filter_fft_input: &mut Vec<Complex<f32>>,
        chunk_output: &mut Vec<Complex<f32>>,
        channels: i32,
    ) -> Self {
        Self {
            forward_plan: plan_helper(
                chunk_fft_input,
                chunk_fft_output,
                &channels,
                &(CHUNK_SIZE as i32),
                FftList::Input,
            ),
            reverse_plan: plan_helper(
                chunk_fft_output,
                filter_fft_input,
                &channels,
                &(CHUNK_SIZE as i32),
                FftList::Filter,
            ),
            down_convert_plan: plan_helper(
                filter_fft_input,
                chunk_output,
                &channels,
                &(CHUNK_SIZE as i32),
                FftList::Downconvert,
            ),
            channels,
        }
    }
}

impl<const TWICE_TAPS: usize, const CHUNK_SIZE: usize> Channelizer<TWICE_TAPS, CHUNK_SIZE> {
    pub fn new(channels: usize) -> Self {
        Self {
            fft: FftPlanner::new().plan_fft_inverse(channels),
            coeff: create_filter::<TWICE_TAPS>(channels),
            conv_coeff: create_filter_chunk::<TWICE_TAPS, CHUNK_SIZE>(channels),
            state: vec![Ring::<Complex<f32>, CHUNK_SIZE>::new(); channels / 2],
            scratch: vec![Complex::zero(); channels],
            chunk_fft_input: vec![Complex::zero(); (CHUNK_SIZE * channels / 2)],
            chunk_fft_output: vec![Complex::zero(); (CHUNK_SIZE * channels)],
            filter_fft_input: vec![Complex::zero(); (CHUNK_SIZE * channels)],
            chunk_output: vec![Complex::zero(); (CHUNK_SIZE * channels)],
            channels,
        }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Add a single slice of channels to the state of this channelizer.
    ///
    /// `add` will only take the first [`channels`] divded by two number of samples from the given
    /// slice. Any additional samples will be ignored. `add` returns the total number of samples
    /// taken from the given slice.
    ///
    /// # Panics
    /// If the length of the given sample slice isn't greater than the number of channels divided by
    /// two, this call will panic. This call is only expected to add a single slice at a time.
    ///
    /// [`channels`]: Self::channels()
    #[inline]
    pub fn add(&mut self, samples: &[Complex<f32>]) -> usize {
        assert!(samples.len() >= self.channels / 2);
        self.state
            .iter_mut()
            .zip(samples.iter().take(self.channels / 2).rev())
            .for_each(|(ring, sample)| ring.add(*sample));

        self.channels / 2
    }

    /// Produce a channelizer slice from this channelizer's current state
    ///
    /// The given output slice is expected to be at least of size equal to [`channels`]. Any
    /// additional space in the output slice is unused. `process` will return the number of
    /// locations modified by the call.
    ///
    /// # Panics
    /// `process` will panic if the length of the output is less than [`channels`].
    ///
    /// [`channels`]: Self::channels()
    pub fn process(&mut self, output: &mut [Complex<f32>]) -> usize {
        output[..self.channels]
            .iter_mut()
            .zip(self.state.iter().chain(self.state.iter()))
            .zip(self.coeff.iter())
            .for_each(|((out, ring), coeff)| {
                *out = ring
                    .inner_iter()
                    .zip(coeff.iter())
                    .fold(Complex::zero(), |accum, (state, coeff)| {
                        accum + state * coeff
                    })
            });
        self.fft
            .process_with_scratch(&mut output[..self.channels], &mut self.scratch);

        self.channels
    }

    pub fn dump_state(&mut self) {
        self.state.iter_mut().enumerate().for_each(|(ind, ring)| {
            ring.inner_iter().enumerate().for_each(|(ind_, item)| {
                self.chunk_fft_input[ind * CHUNK_SIZE + ind_] =
                    (*item) / (Complex::new((CHUNK_SIZE * self.channels) as f32, 0.0))
            })
        })
    }
    pub fn process_all(
        &mut self,
        plans: ChannelizationPlans<CHUNK_SIZE>,
    ) {
        self.dump_state();
        // println!("dumping state: {:?}", now.elapsed());
        
        unsafe { fftwf_execute(plans.forward_plan) };
        self.chunk_fft_output
            .iter_mut()
            .zip(self.conv_coeff.iter())
            .zip(self.chunk_fft_input.iter())
            .for_each(|((out, coeff), inp)| *out = inp * coeff);
        let now = Instant::now();
        // unsafe { fftwf_execute(plans.reverse_plan) };
        unsafe { fftwf_execute(plans.down_convert_plan) };
        println!("other ffts: {:?}", now.elapsed());

    }

    /// Resets the state of this channelizer
    pub fn reset(&mut self) {
        for ring in self.state.iter_mut() {
            ring.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::channel;

    use super::*;
    use rayon::prelude::*;

    const CHANNELS: usize = 4096;
    const TWICE_TAPS: usize = 24;
    const INPUT_SIGNAL: [Complex<f32>; CHANNELS / 2] = [Complex::new(1.0, 0.0); CHANNELS / 2];
    const CHUNK_SIZE: usize = 64;

    #[test]
    fn process() {
        let mut channelizer = Channelizer::<TWICE_TAPS, CHUNK_SIZE>::new(CHANNELS);
        let mut plans = ChannelizationPlans::<CHUNK_SIZE>::new(
            &mut channelizer.chunk_fft_input,
            &mut channelizer.chunk_fft_output,
            &mut channelizer.filter_fft_input,
            &mut channelizer.chunk_output,
            channelizer.channels as i32,
        );
        // let mut output = vec![Complex::zero(); CHANNELS];

        let now = std::time::Instant::now();
        for _ in 0..CHUNK_SIZE {
            channelizer.add(&INPUT_SIGNAL);
            // channelizer.process(&mut output);
        }
        // let t0 = now.elapsed().as_secs_f32();
        // println!("time taken to add: {:?}", t0);
        // let now2 = std::time::Instant::now();
        channelizer.process_all(plans);
        // let t = now2.elapsed().as_secs_f32();
        // println!("time to process given chunk: {:?}", t);
        // println!("throughput: {:?}", ((CHUNK_SIZE * CHANNELS / 2) as f32) / t);
        println!("sample output: {:?}", &channelizer.chunk_output[..2]);
    }

    // #[test]
    // fn par_process() {
    //     const INNER_LOOPS: usize = 10_000;
    //     const CHUNKS: usize = 50;

    //     let mut output = vec![[Complex::zero(); CHANNELS]; CHUNKS];
    //     let mut channelizers = vec![Channelizer::<TWICE_TAPS>::new(CHANNELS); output.len()];

    //     let now = std::time::Instant::now();
    //     output
    //         .par_iter_mut()
    //         .zip(channelizers.par_iter_mut())
    //         .for_each(|(output, channelizer)| {
    //             for _ in 0..INNER_LOOPS {
    //                 channelizer.add(&INPUT_SIGNAL);
    //                 channelizer.process(output);
    //             }
    //         });
    //     println!(
    //         "time to process {:?} slices: {:?}",
    //         INNER_LOOPS * CHUNKS,
    //         now.elapsed()
    //     );
    //     println!("using {:?} taps, {:?} channels", TWICE_TAPS / 2, CHANNELS);
    // }

    // #[test]
    // fn reset() {
    //     let mut channelizer = Channelizer::<TWICE_TAPS>::new(CHANNELS);
    //     let mut output = vec![Complex::zero(); CHANNELS];

    //     channelizer.add(&INPUT_SIGNAL);
    //     channelizer.process(&mut output);

    //     let copy = output.clone();

    //     channelizer.reset();
    //     channelizer.add(&INPUT_SIGNAL);
    //     channelizer.process(&mut output);

    //     assert_eq!(copy, output);
    // }
}
