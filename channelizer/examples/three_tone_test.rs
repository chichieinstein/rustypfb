pub use channelizer::{sinc, ChunkChannelizer};
use num::{Complex, Zero};
use num_complex::Complex32;
pub use rustdevice::{compute_bessel, DevicePtr};

use std::{
    f32::consts::PI,
    io::{Read, Write},
};

fn main() {
    let nch = 1024;
    let ntaps = 32;
    let nslice = 512;
    let float_taps = (-ntaps / 2) as f32;
    let chann_float = nch as f32;
    let chann_proto = ntaps as f32;
    let kbeta = 10 as f32;
    let nsamples = nch * nslice;

    /*
     * Setup the channelizer prototype filter parameters
     */
    let mut filter: Vec<f32> = (0..nch * ntaps)
        .map(|x| {
            let y = x as f32;
            let arg = float_taps + (y + 1.0) / chann_float;
            let darg = (2.0 * y) / (chann_float * chann_proto) - 1.0;
            let carg = kbeta * (1.0 - darg * darg).sqrt();
            (compute_bessel(carg) / compute_bessel(kbeta)) * sinc(arg)
        })
        .collect();

    let fc1: f32 = 0.5e6;
    let fc2: f32 = 1.25e6;
    let fc3: f32 = 0.85e6;
    let a2_db: f32 = -100.0;
    let fs: f32 = 100e6;

    let mut chann_obj = ChunkChannelizer::new(filter.as_mut_slice(), ntaps, nch, nslice);

    // Setup the output buffer
    let mut channelized_output_buffer = DevicePtr::<Complex<f32>>::new(nch * nslice);

    // Setup the input vector
    let mut input_vec = vec![Complex::zero() as Complex<f32>; (nch * nslice / 2) as usize];

    input_vec.iter_mut().enumerate().for_each(|(ind, elem)| {
        *elem = Complex::new(
            (2.0 * PI * fc1 * (ind as f32) / fs).cos(),
            (-2.0 * PI * fc1 * (ind as f32) / fs).sin(),
        ) + Complex::new(
            (2.0 * PI * fc3 * (ind as f32) / fs).cos(),
            (-2.0 * PI * fc3 * (ind as f32) / fs).sin(),
        ) + Complex::new(
            (2.0 * PI * fc2 * (ind as f32) / fs).cos() * 10.0f32.powf(a2_db),
            (-2.0 * PI * fc2 * (ind as f32) / fs).sin() * 10.0f32.powf(a2_db),
        )
    });

    let mut tone_file = std::fs::File::create("tones.32cf").unwrap();

    let tone_outp_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut input_vec);

    let _ = tone_file.write_all(tone_outp_slice);
}
