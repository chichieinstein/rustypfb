pub use channelizer::{sinc, ChunkChannelizer};
use num::{Complex, Zero};
pub use rustdevice::{compute_bessel, DevicePtr};
use uhd::{StreamArgs, StreamCommand, StreamCommandType, StreamTime, TuneRequest, Usrp};

use std::{
    f32::consts::PI,
    io::{ Write},
    sync::{Arc, Mutex},
    cmp,
};

const DEFAULT_CHANNEL: usize = 0;

const TX_ANTENNA: &str = "TX/RX";
const TX_ADDR: &str = "192.168.101.20";
const TX_USRP_TYPE: &str = "x300";
const TX_MCR: usize = 200_000_000;
const TX_NORM_GAIN: f64 = 0.2;

const RX_ANTENNA: &str = "TX/RX";
const RX_ADDR: &str = "192.168.101.16";
const RX_USRP_TYPE: &str = "n3xx";
const RX_MCR: usize = 200_000_000;
const RX_NORM_GAIN: f64 = 0.6;

fn main() {
    let nch = 1024;
    let ntaps = 32;
    let nslice = 512;
    let float_taps = (-(ntaps as i64) / 2) as f32;
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

    let a2_db: f32 = -9.5;
    let mut amplitude: f32 = 0.8;
    let fs: f32 = 5e6;
    let rx_fs: f32 = 100e6;

    let mut chann_obj = ChunkChannelizer::new(filter.as_mut_slice(), ntaps, nch, nslice);

    // Setup the baseband vector
    let mut baseband = vec![Complex::zero() as Complex<f32>; (nch * nslice / 2) as usize];

    baseband.iter_mut().enumerate().for_each(|(ind, elem)| {
        *elem = Complex::new(
            amplitude * (2.0 * PI * fc1 * (ind as f32) / fs).cos(),
            amplitude * (-2.0 * PI * fc1 * (ind as f32) / fs).sin(),
        ) + Complex::new(
            amplitude * (2.0 * PI * fc3 * (ind as f32) / fs).cos(),
            amplitude * (-2.0 * PI * fc3 * (ind as f32) / fs).sin(),
        ) 
        + Complex::new(
            amplitude * (2.0 * PI * fc2 * (ind as f32) / fs).cos() * 10.0f32.powf(a2_db/10.0),
            amplitude * (-2.0 * PI * fc2 * (ind as f32) / fs).sin() * 10.0f32.powf(a2_db/10.0),
        )
    });

    let mut synth_baseband = vec![Complex::zero() as Complex<f32>; (nch * nslice / 2) as usize];

    synth_baseband.iter_mut().enumerate().for_each(|(ind, elem)| {
        *elem = Complex::new(
            amplitude * (2.0 * PI * fc1 * (ind as f32) / rx_fs).cos(),
            amplitude * (-2.0 * PI * fc1 * (ind as f32) / rx_fs).sin(),
        ) + Complex::new(
            amplitude * (2.0 * PI * fc3 * (ind as f32) / rx_fs).cos(),
            amplitude * (-2.0 * PI * fc3 * (ind as f32) / rx_fs).sin(),
        ) 
        + Complex::new(
            amplitude * (2.0 * PI * fc2 * (ind as f32) / rx_fs).cos() * 10.0f32.powf(a2_db/10.0),
            amplitude * (-2.0 * PI * fc2 * (ind as f32) / rx_fs).sin() * 10.0f32.powf(a2_db/10.0),
        )
    });

    let max_amp = baseband.iter()
        .map(|c| c.norm())
        .fold(0_f32, |max, amp| if amp > max { amp } else { max });

    let mut normalized_baseband: Vec<_> = baseband.iter()
        .map(|c| 0.8 * c / max_amp)
        .collect();

    let max_norm_amp = normalized_baseband.iter()
        .map(|c| c.norm())
        .fold(0_f32, |max, amp| if amp > max { amp } else { max });


    println!("Max amplitude: {}", max_norm_amp);

    let radio_center_freq = 3.0e9;

    // Create a tx USRP object
    let args = format!(
        "addr={},type={},master_clock_rate={}",
        TX_ADDR, TX_USRP_TYPE, TX_MCR
    );
    let tx_channel = DEFAULT_CHANNEL;
    let mut tx = match Usrp::open(args.as_str()) {
        Ok(usrp) => usrp,
        Err(e) => panic!("failed to create USRP: {}", e),
    };

    // Create a rx USRP object
    let args = format!(
        "addr={},type={},master_clock_rate={}",
        RX_ADDR, RX_USRP_TYPE, RX_MCR
    );
    let rx_channel = DEFAULT_CHANNEL;
    let mut rx = match Usrp::open(args.as_str()) {
        Ok(usrp) => usrp,
        Err(e) => panic!("failed to create USRP: {}", e),
    };

    let rx_gain = linear_map(RX_NORM_GAIN, 0.0, 1.0, 0.0, 60.0);
    let tx_gain = linear_map(TX_NORM_GAIN, 0.0, 1.0, 0.0, 60.0);

    // Configure the tx USRP
    tx.set_tx_antenna(TX_ANTENNA, tx_channel)
        .expect("Failed to set TX antenna");
    tx.set_tx_sample_rate(fs.into(), tx_channel)
        .expect("Failed to set TX sample rate");
    tx.set_tx_frequency(&TuneRequest::with_frequency(radio_center_freq), tx_channel)
        .expect("Failed to set TX frequency");
    tx.set_tx_gain(tx_gain, tx_channel, "")
        .expect("Failed to set TX gain");

    // Configure the rx USRP
    rx.set_rx_antenna(RX_ANTENNA, rx_channel)
        .expect("Failed to set RX antenna");
    rx.set_rx_sample_rate(rx_fs.into(), rx_channel)
        .expect("Failed to set RX sample rate");
    rx.set_rx_frequency(&TuneRequest::with_frequency(radio_center_freq), rx_channel)
        .expect("Failed to set RX frequency");
    rx.set_rx_gain(rx_gain, rx_channel, "")
        .expect("Failed to set RX gain");

    // used to synchronize the transmission and reception threads
    let tx_running = Arc::new(Mutex::new(false));
    let rx_running = tx_running.clone();

    // spawn a new thread to continously receive samples
    let rx_thrd_handle = std::thread::spawn(move || {
        let mut raw_input_vec = vec![Complex::zero() as Complex<f32>; 0];

        // wait for the transmission to start
        let mut has_ran = false;
        while !has_ran {
            while rx_running.lock().unwrap().eq(&true) {
                has_ran = true;
                let num_rx_samps = nsamples as usize;
                let mut receiver = rx
                    .get_rx_stream(&StreamArgs::<Complex<f32>>::new("sc16"))
                    .expect("Failed to get RX stream");

                let mut rx_binding = vec![Complex::zero() as Complex<f32>; num_rx_samps];
                let mut rx_buffer = Box::new(rx_binding.as_mut_slice());

                receiver
                    .send_command(&StreamCommand {
                        command_type: StreamCommandType::CountAndDone(rx_buffer.len() as u64),
                        time: StreamTime::Now,
                    })
                    .expect("Failed to start RX stream");

                let _meta = receiver
                    .receive_simple(rx_buffer.as_mut())
                    .expect("Failed to receive samples");

                // append these samples to the raw input vector
                raw_input_vec.extend(rx_binding.drain(..))
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        return raw_input_vec;
    });

    // spawn a new thread to transmit samples
    let tx_thrd_handle = std::thread::spawn(move || {
        let mut sender = tx
            .get_tx_stream(&StreamArgs::<Complex<f32>>::new("sc16"))
            .expect("Failed to get TX stream");

        let mut tx_buffer = Box::new(normalized_baseband.as_mut_slice());

        *tx_running.lock().unwrap() = true;
        for _ in 0..10 {
            let _meta = sender
                .transmit_simple(tx_buffer.as_mut())
                .expect("Failed to transmit");
        }
        *tx_running.lock().unwrap() = false;
    });

    let raw_input_vec = rx_thrd_handle.join().unwrap();
    tx_thrd_handle.join().unwrap();

    println!("\nFinished transmission and reception");
    println!("Received {} samples", raw_input_vec.len());

    let mut input_vec = raw_input_vec[0..(nch * nslice / 2) as usize].to_vec();

    println!("Processing {} samples", input_vec.len());

    let path = std::path::Path::new("./iq/tones.32cf");
    let prefix = path.parent().unwrap();
    std::fs::create_dir_all(prefix).unwrap();
    let mut tone_file = std::fs::File::create(path).unwrap();
    let tone_outp_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut input_vec);
    let _ = tone_file.write_all(tone_outp_slice);

    let synthetic_path = std::path::Path::new("./iq/synthetic.32cf");
    let mut synthetic_file = std::fs::File::create(synthetic_path).unwrap();
    let synthetic_outp_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut synth_baseband);
    let _ = synthetic_file.write_all(synthetic_outp_slice);

    // OTA Channogram
    let mut inp_vec_float = vec![0.0 as f32; (nch * nslice) as usize];
    let inp_vec_cmp: &[f32] = bytemuck::cast_slice(&input_vec);
    inp_vec_float[0..(nch * nslice) as usize].clone_from_slice(inp_vec_cmp);

    // Setup the output buffer
    let mut output_buffer = DevicePtr::<Complex<f32>>::new(nch * nslice);

    // Setup the CPU output buffer
    let mut output_cpu = vec![Complex::<f32>::zero(); (nch * nslice) as usize];

    chann_obj.process(&mut inp_vec_float, &mut output_buffer);
    output_buffer.dump(&mut output_cpu);

    let mut chann_file = std::fs::File::create("./iq/tone_channelized.32cf").unwrap();
    let tone_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut output_cpu);
    let _ = chann_file.write_all(tone_slice);

    // Synthetic Channogram
    let mut synth_inp_vec_float = vec![0.0 as f32; (nch * nslice) as usize];
    let synth_inp_vec_cmp: &[f32] = bytemuck::cast_slice(&synth_baseband);
    synth_inp_vec_float[0..(nch * nslice) as usize].clone_from_slice(synth_inp_vec_cmp);

    // Setup the output buffer
    let mut synth_output_buffer = DevicePtr::<Complex<f32>>::new(nch * nslice);

    // Setup the CPU output buffer
    let mut synth_output_cpu = vec![Complex::<f32>::zero(); (nch * nslice) as usize];

    chann_obj.process(&mut synth_inp_vec_float, &mut synth_output_buffer);
    synth_output_buffer.dump(&mut synth_output_cpu);

    let mut synth_chann_file = std::fs::File::create("./iq/synthetic_channelized.32cf").unwrap();
    let synth_tone_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut synth_output_cpu);
    let _ = synth_chann_file.write_all(synth_tone_slice);

}

use std::ops::{Add, Div, Mul, Sub};
fn linear_map<T>(x: T, in_min: T, in_max: T, out_min: T, out_max: T) -> T
where
    T: PartialOrd + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Add<Output = T> + Copy,
{
    if in_max < in_min {
        panic!("in_max must be greater than in_min");
    }
    if x < in_min {
        return out_min;
    }
    if x > in_max {
        return out_max;
    }
    (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
}
