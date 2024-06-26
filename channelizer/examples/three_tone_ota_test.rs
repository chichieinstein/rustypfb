pub use channelizer::{sinc, ChunkChannelizer};
use num::{Complex, Zero};
pub use rustdevice::{compute_bessel, DevicePtr};
use uhd::{StreamArgs, StreamCommand, StreamCommandType, StreamTime, TuneRequest, Usrp};

use std::{
    cmp,
    f32::consts::PI,
    io::Write,
    sync::{Arc, Mutex},
};

const TX_MCR: usize = 200_000_000;
const RX_MCR: usize = 200_000_000;

fn main() {
    let main_prog = clap::Command::new("channogram")
        .version("unknown")
        .about("A tool for generating channograms from IQ samples collected ota")
        .subcommands([clap::Command::new("run").args([
            // TX USRP parameters
            clap::Arg::new("tx_antenna")
                .long("tx_antenna")
                .value_name("TX_ANTENNA")
                .help("The antenna to use for transmission")
                .default_value("TX/RX"),
            clap::Arg::new("tx_channel")
                .long("tx_channel")
                .value_name("TX_CHANNEL")
                .help("The channel to use for transmission")
                .default_value("0"),
            clap::Arg::new("tx_addr")
                .long("tx_addr")
                .value_name("TX_ADDR")
                .help("The address of the USRP to use for transmission")
                .default_value("192.168.101.20"),
            clap::Arg::new("tx_usrp_type")
                .long("tx_usrp_type")
                .value_name("TX_USRP_TYPE")
                .help("The type of USRP to use for transmission")
                .default_value("x300"),
            clap::Arg::new("tx_norm_gain")
                .long("tx_norm_gain")
                .value_name("TX_NORM_GAIN")
                .help("The normalized gain to use for transmission")
                .default_value("0.2"),
            // RX USRP parameters
            clap::Arg::new("rx_antenna")
                .long("rx_antenna")
                .value_name("RX_ANTENNA")
                .help("The antenna to use for reception")
                .default_value("TX/RX"),
            clap::Arg::new("rx_channel")
                .long("rx_channel")
                .value_name("RX_CHANNEL")
                .help("The channel to use for reception")
                .default_value("0"),
            clap::Arg::new("rx_addr")
                .long("rx_addr")
                .value_name("RX_ADDR")
                .help("The address of the USRP to use for reception")
                .default_value("192.168.101.16"),
            clap::Arg::new("rx_usrp_type")
                .long("rx_usrp_type")
                .value_name("RX_USRP_TYPE")
                .help("The type of USRP to use for reception")
                .default_value("n3xx"),
            clap::Arg::new("rx_norm_gain")
                .long("rx_norm_gain")
                .value_name("RX_NORM_GAIN")
                .help("The normalized gain to use for reception")
                .default_value("0.6"),
        ])]);

    let config = main_prog.clone().get_matches();
    if let Some(run_config) = config.subcommand_matches("run") {
        let tx_antenna = run_config.get_one::<String>("tx_antenna").unwrap();
        let tx_channel_str = run_config.get_one::<String>("tx_channel").unwrap();
        let tx_channel = tx_channel_str.parse::<usize>().unwrap();
        let tx_addr = run_config.get_one::<String>("tx_addr").unwrap();
        let tx_usrp_type = run_config.get_one::<String>("tx_usrp_type").unwrap();
        let tx_norm_gain_str = run_config.get_one::<String>("tx_norm_gain").unwrap();
        let tx_norm_gain = tx_norm_gain_str.parse::<f64>().unwrap();

        let rx_antenna = run_config.get_one::<String>("rx_antenna").unwrap();
        let rx_channel_str = run_config.get_one::<String>("rx_channel").unwrap();
        let rx_channel = rx_channel_str.parse::<usize>().unwrap();
        let rx_addr = run_config.get_one::<String>("rx_addr").unwrap();
        let rx_usrp_type = run_config.get_one::<String>("rx_usrp_type").unwrap();
        let tx_norm_gain_str = run_config.get_one::<String>("rx_norm_gain").unwrap();
        let rx_norm_gain = tx_norm_gain_str.parse::<f64>().unwrap();

        print_config(run_config);

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

        let a2_db: f32 = -17.5;
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
            ) + Complex::new(
                amplitude * (2.0 * PI * fc2 * (ind as f32) / fs).cos() * 10.0f32.powf(a2_db / 10.0),
                amplitude
                    * (-2.0 * PI * fc2 * (ind as f32) / fs).sin()
                    * 10.0f32.powf(a2_db / 10.0),
            )
        });

        let mut synth_baseband = vec![Complex::zero() as Complex<f32>; (nch * nslice / 2) as usize];

        synth_baseband
            .iter_mut()
            .enumerate()
            .for_each(|(ind, elem)| {
                *elem = Complex::new(
                    amplitude * (2.0 * PI * fc1 * (ind as f32) / rx_fs).cos(),
                    amplitude * (-2.0 * PI * fc1 * (ind as f32) / rx_fs).sin(),
                ) + Complex::new(
                    amplitude * (2.0 * PI * fc3 * (ind as f32) / rx_fs).cos(),
                    amplitude * (-2.0 * PI * fc3 * (ind as f32) / rx_fs).sin(),
                ) + Complex::new(
                    amplitude
                        * (2.0 * PI * fc2 * (ind as f32) / rx_fs).cos()
                        * 10.0f32.powf(a2_db / 10.0),
                    amplitude
                        * (-2.0 * PI * fc2 * (ind as f32) / rx_fs).sin()
                        * 10.0f32.powf(a2_db / 10.0),
                )
            });

        let max_amp =
            baseband
                .iter()
                .map(|c| c.norm())
                .fold(0_f32, |max, amp| if amp > max { amp } else { max });

        let mut normalized_baseband: Vec<_> = baseband.iter().map(|c| 0.8 * c / max_amp).collect();

        let radio_center_freq = 3.0e9;

        // Create a tx USRP object
        let args = format!(
            "addr={},type={},master_clock_rate={}",
            tx_addr, tx_usrp_type, TX_MCR
        );
        let mut tx = match Usrp::open(args.as_str()) {
            Ok(usrp) => usrp,
            Err(e) => panic!("failed to create USRP: {}", e),
        };

        // Create a rx USRP object
        let args = format!(
            "addr={},type={},master_clock_rate={}",
            rx_addr, rx_usrp_type, RX_MCR
        );
        let mut rx = match Usrp::open(args.as_str()) {
            Ok(usrp) => usrp,
            Err(e) => panic!("failed to create USRP: {}", e),
        };

        let rx_gain = linear_map(rx_norm_gain, 0.0, 1.0, 0.0, 60.0);
        let tx_gain = linear_map(tx_norm_gain, 0.0, 1.0, 0.0, 60.0);

        // Configure the tx USRP
        tx.set_tx_antenna(tx_antenna, tx_channel)
            .expect("Failed to set TX antenna");
        tx.set_tx_sample_rate(fs.into(), tx_channel)
            .expect("Failed to set TX sample rate");
        tx.set_tx_frequency(&TuneRequest::with_frequency(radio_center_freq), tx_channel)
            .expect("Failed to set TX frequency");
        tx.set_tx_gain(tx_gain, tx_channel, "")
            .expect("Failed to set TX gain");

        // Configure the rx USRP
        rx.set_rx_antenna(rx_antenna, rx_channel)
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

        let path = std::path::Path::new("./iq/ota_tones.32cf");
        let prefix = path.parent().unwrap();
        std::fs::create_dir_all(prefix).unwrap();
        let mut tone_file = std::fs::File::create(path).unwrap();
        let tone_outp_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut input_vec);
        let _ = tone_file.write_all(tone_outp_slice);

        let synthetic_path = std::path::Path::new("./iq/synthetic_tones.32cf");
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

        let mut chann_file = std::fs::File::create("./iq/ota_tones_channelized.32cf").unwrap();
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

        let mut synth_chann_file =
            std::fs::File::create("./iq/synthetic_tones_channelized.32cf").unwrap();
        let synth_tone_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut synth_output_cpu);
        let _ = synth_chann_file.write_all(synth_tone_slice);
    }
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

fn print_config(run_config: &clap::ArgMatches) {
    println!("\n-------------------------");
    println!("Transmitter Configuration");
    println!(
        "Antenna: {}",
        run_config.get_one::<String>("tx_antenna").unwrap()
    );
    println!(
        "Channel: {}",
        run_config.get_one::<String>("tx_channel").unwrap()
    );
    println!(
        "Address: {}",
        run_config.get_one::<String>("tx_addr").unwrap()
    );
    println!(
        "USRP Type: {}",
        run_config.get_one::<String>("tx_usrp_type").unwrap()
    );
    println!(
        "Normalized Gain: {}\n",
        run_config.get_one::<String>("tx_norm_gain").unwrap()
    );

    println!("Receiver Configuration");
    println!(
        "Antenna: {}",
        run_config.get_one::<String>("rx_antenna").unwrap()
    );
    println!(
        "Channel: {}",
        run_config.get_one::<String>("rx_channel").unwrap()
    );
    println!(
        "Address: {}",
        run_config.get_one::<String>("rx_addr").unwrap()
    );
    println!(
        "USRP Type: {}",
        run_config.get_one::<String>("rx_usrp_type").unwrap()
    );
    println!(
        "Normalized Gain: {}",
        run_config.get_one::<String>("rx_norm_gain").unwrap()
    );
    println!("-------------------------\n");
}
