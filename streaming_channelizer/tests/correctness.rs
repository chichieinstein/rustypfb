use num::{Complex, Zero};
use streaming_channelizer::StreamingMaximalChannelizer;
use std::f32::consts::PI;

// #[test]
// fn correctness_test() {
//     // Setting up constants for the Channelizer
//     const CHANNELS: usize = 1024;
//     const CHUNK_SIZE: usize = 8192 * 2;
//     const HOP_SIZE: usize = 1024;
//     const TAPS: usize = 16;

//     // Create the Channelizer
//     let mut channelizer =
//         StreamingMaximalChannelizer::<CHUNK_SIZE, TAPS, HOP_SIZE>::new(CHANNELS as i32, None);

//     // Initialize samples
//     let samples: Vec<Complex<f32>> = (0..CHUNK_SIZE)
//         .map(|x| Complex::new((2.0*PI*(x as f32)).cos(), (2.0*PI*(x as f32)).sin()))
//         .collect();

//     let mut output = vec![Complex::<f32>::zero(); CHUNK_SIZE];

//     channelizer.add(&samples);
//     channelizer.process(&mut output);

//     for ind in 0..100
//     {
//         println!("{}\n", output[ind]);
//     }
// }
