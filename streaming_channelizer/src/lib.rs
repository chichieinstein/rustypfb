use bessel_fun_sys::bessel_func;
use num::{Complex, Zero};
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Default proto-type filter based on the Kaiser window, with kbeta=10.0
fn kaiser_fn(ind: usize, nchannel: usize, nproto: usize) -> f32 {
    let ind_arg = ind as f32;
    let arg = -((nproto / 2) as f32) + (ind_arg + 1.0) / (nchannel as f32);
    let darg = (2.0 * ind_arg) / ((nchannel * nproto) as f32) - 1.0;
    let carg = 10.0 * (1.0 - darg * darg).sqrt();
    (unsafe { bessel_func(carg) }) / (unsafe { bessel_func(10.0) })
        * (if arg != 0.0 { arg.sin() / arg } else { 1.0 })
}

/// The prototype filter to be used in the Polyphase channelizer.
/// The fn_ptr argument is an option type that optionally contains a function pointer that points to a function taking as input
/// the array index (as usize), number of channels (as usize) and number of taps per channel (as usize) and returns a float value
///
/// Thus, the function should have a signature
/// fn func(usize, usize, usize) -> f32
///
/// If the fn_ptr argument is None, then the proto-type filter defaults to the product of Kaiser window and sinc with the given (ind, nchannels, ntaps) argument.
fn create_filter<const TAPS: usize>(
    channels: usize,
    slice: usize,
    fn_ptr: Option<fn(usize, usize, usize) -> f32>,
) -> Vec<Complex<f32>> {
    let chann_fn = match fn_ptr {
        None => kaiser_fn,
        Some(fn_ptr) => fn_ptr,
    };
    let mut result = vec![Complex::<f32>::zero(); channels * slice];
    let taps = TAPS;
    let mut fft_planner = FftPlanner::new();
    let fft = fft_planner.plan_fft_forward(slice);
    for chann_id in 0..channels {
        let buffer = &mut result[chann_id * slice..(chann_id + 1) * slice];
        for tap_id in 0..taps {
            let ind = tap_id * channels + chann_id;
            buffer[tap_id] = Complex::<f32>::new(chann_fn(ind, channels, taps), 0.0);
        }
        fft.process(buffer);
    }
    result
}

/// This is the internal state of the Streaming Channelizer.
/// The generic argument CHUNK_SIZE refers to the size of input that the channelizer processes in one shot.
/// The generic argument HOP_SIZE controls the amount of overlap that the user wants to maintain between successive inputs.
/// CHUNK_SIZE - HOP_SIZE samples will be common between successive input chunks.
/// The first call to add() after calling reset() on a ChannState instance will expect CHUNK_SIZE samples, and all memory of previous chunks will be lost on the output size.
#[derive(Copy, Clone, Debug)]
struct ChannState<T, const CHUNK_SIZE: usize, const HOP_SIZE: usize> {
    buffer: [T; CHUNK_SIZE],
    reset: bool,
}

impl<T: Default + Copy, const CHUNK_SIZE: usize, const HOP_SIZE: usize>
    ChannState<T, CHUNK_SIZE, HOP_SIZE>
{
    fn new() -> Self {
        Self {
            buffer: [T::default(); CHUNK_SIZE],
            reset: true,
        }
    }

    #[inline]
    fn add(&mut self, samples: &[T]) {
        if self.reset {
            self.buffer.clone_from_slice(samples);
            self.reset = false;
        } else {
            self.buffer.rotate_left(HOP_SIZE);
            self.buffer[CHUNK_SIZE - HOP_SIZE..].clone_from_slice(&samples[..HOP_SIZE]);
        }
    }

    fn reset(&mut self) {
        self.reset = true;
    }
}

/// The streaming Polyphase filter bank that acts as a maximally decimating channelizer.
/// CHUNK_SIZE is the size of input chunks.
/// The TAPS argument sets the number of prototype filter taps per channel.
/// The HOP_SIZE argument sets the overlap between successive chunks.
/// 
/// HOP_SIZE >= channels. If these are equal, we get slice by slice update in each channel.
pub struct StreamingMaximalChannelizer<
    const CHUNK_SIZE: usize,
    const TAPS: usize,
    const HOP_SIZE: usize,
> {
    state: ChannState<Complex<f32>, CHUNK_SIZE, HOP_SIZE>,
    input: [Complex<f32>; CHUNK_SIZE],
    filter_output: Vec<Complex<f32>>,
    initial_fft_scratch: Vec<Complex<f32>>,
    downconvert_fft_scratch: Vec<Complex<f32>>,
    channels: i32,
    filter: Vec<Complex<f32>>,
    output: Vec<Complex<f32>>,
    output_scratch: Vec<Complex<f32>>,
    fft_initial: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    fft_downconvert: Arc<dyn Fft<f32>>,
}

impl<const CHUNK_SIZE: usize, const TAPS: usize, const HOP_SIZE: usize>
    StreamingMaximalChannelizer<CHUNK_SIZE, TAPS, HOP_SIZE>
{
    /// Create a Channelizer instance. Arguments are channels, and an optional function pointer that computes the prototype filter coefficients.
    /// Default function is the Kaiser prototype filter.
    pub fn new(channels: i32, proto_filter: Option<fn(usize, usize, usize) -> f32>) -> Self {
        let state = ChannState::<Complex<f32>, CHUNK_SIZE, HOP_SIZE>::new();
        let input = [Complex::<f32>::zero(); CHUNK_SIZE];
        let filter_output = vec![Complex::<f32>::zero(); CHUNK_SIZE];
        let initial_fft_scratch = vec![Complex::<f32>::zero(); CHUNK_SIZE / (channels as usize)];
        let downconvert_fft_scratch = vec![Complex::<f32>::zero(); channels as usize];
        let output_scratch = vec![Complex::<f32>::zero(); channels as usize];
        let output = vec![Complex::<f32>::zero(); CHUNK_SIZE];
        let mut plan = FftPlanner::new();

        StreamingMaximalChannelizer {
            state,
            input,
            filter_output,
            filter: create_filter::<TAPS>(
                channels as usize,
                CHUNK_SIZE / (channels as usize),
                proto_filter,
            ),
            channels,
            initial_fft_scratch,
            downconvert_fft_scratch,
            fft_initial: plan.plan_fft_forward(CHUNK_SIZE / (channels as usize)),
            fft_inverse: plan.plan_fft_inverse(CHUNK_SIZE / (channels as usize)),
            fft_downconvert: plan.plan_fft_inverse(channels as usize),
            output_scratch,
            output,
        }
    }

    /// Update the state of the Channelizer.
    pub fn add(&mut self, samples: &[Complex<f32>]) {
        self.state.add(samples);
    }

    /// Reset the state of the Channelier in preparation for a new stream.
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Process the input. The Channelizer is maximally decimating, and therefore,
    /// the number of input samples is the same as the number of output samples, and 
    /// all channels are disjoint.
    /// 
    /// The main aim of the Streaming Channelizer is to minimize the latency between input and 
    /// output, therefore, CHUNK_SIZE in typical cases will be small : less than 50K samples at a time.
    /// With such small number of samples, data parallelism (either with Rayon, or with a custom Thread Pool) cannot be achieved, as the overhead of 
    /// wrapping inputs in Mutexes and then obtaining locks in parallel threads is much larger than any 
    /// gain in parallelism. Therefore, this function is restricted to one core.
    pub fn process(&mut self, output: &mut [Complex<f32>]) {
        let nslice = (CHUNK_SIZE) / (self.channels as usize);
        self.input
            .chunks_mut(nslice)
            .zip((0..self.channels).map(|ind| {
                self.state.buffer[ind as usize..]
                    .iter()
                    .step_by((self.channels) as usize)
            }))
            .for_each(|(x, y)| {
                y.into_iter().zip(x).for_each(|(yl, xl)| *xl = *yl);
            });
        self.input
            .chunks_mut(CHUNK_SIZE / (self.channels as usize))
            .for_each(|chunk| {
                self.fft_initial
                    .process_with_scratch(chunk, &mut self.initial_fft_scratch)
            });
        self.filter
            .iter()
            .zip(self.input.iter())
            .zip(self.filter_output.iter_mut())
            .for_each(|((filter, input), output)| *output = input * filter);

        self.filter_output
            .chunks_mut(CHUNK_SIZE / (self.channels as usize))
            .for_each(|chunk| {
                self.fft_inverse
                    .process_with_scratch(chunk, &mut self.initial_fft_scratch);
            });
        (0..nslice).for_each(|ind| {
            self.filter_output[ind..]
                .iter()
                .step_by(nslice as usize)
                .zip(self.output_scratch.iter_mut())
                .for_each(|(y, z)| *z = *y);
            self.fft_downconvert
                .process_with_scratch(&mut self.output_scratch, &mut self.downconvert_fft_scratch);
            self.output[ind..]
                .iter_mut()
                .step_by(nslice as usize)
                .zip(self.output_scratch.iter())
                .for_each(|(o_el, w_elem)| *o_el = *w_elem);
        });
        output.clone_from_slice(&self.output);
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use num::Zero;
    use std::time::Instant;

    const CHANNELS: usize = 1024;
    const CHUNK_SIZE: usize = 8192 * 2;
    const HOP_SIZE: usize = 512;
    const TAPS: usize = 16;

    #[test]
    fn state_timing() {
        let ntimes = 100 as usize;
        let mut channelizer =
            StreamingMaximalChannelizer::<CHUNK_SIZE, TAPS, HOP_SIZE>::new(CHANNELS as i32, None);
        let others = vec![Complex::<f32>::zero(); CHUNK_SIZE];
        let mut output = vec![Complex::<f32>::zero(); CHUNK_SIZE];

        let now = Instant::now();
        for _ in 0..ntimes {
            channelizer.add(&others);
            channelizer.process(&mut output);
        }
        println!("{:?}", now.elapsed().as_nanos() as f32 / (ntimes as f32));
    }
}
