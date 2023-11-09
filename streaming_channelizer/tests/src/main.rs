use streaming_channelizer::StreamChannelizer;
use std::time::Instant;
use num::Complex;


fn main() {
    let channels = 4096;
    let taps = 12;
    let ntimes = 1000;
    let input_buffer = vec![Complex::new(0.0 as f32, 0.0 as f32); channels / 2];
    let mut output_buffer = vec![Complex::new(0.0 as f32, 0.0 as f32); channels];

    let mut chann_obj = StreamChannelizer::new(taps, channels);

    let mut single_slice_latency = 0.0;
    let mut throughput_time = 0.0;

    for _ in 0..ntimes
    {
        let init_inst = Instant::now();
        chann_obj.process(&input_buffer, &mut output_buffer);
        let fin_inst = Instant::now();
        let tot_time = (fin_inst - init_inst).as_secs_f32();
        single_slice_latency += tot_time;
        throughput_time += tot_time;
    }

    println!("Single slice latency : {}, Throughput : {}", single_slice_latency / (ntimes as f32), ((channels / 2 * ntimes) as f32) / throughput_time)
}
