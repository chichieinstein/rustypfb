This is an implementation of a non-maximally decimating polyphase filter bank (PFB), which supports perfect reconstruction. The filter bank has been written to be usable as a Rust crate. The backend is written in CUDA C++ for real time deployment.

In order to check things are running as required, the test written in the root of the ``channelizer`` crate needs to be run. All dependencies are listed in the Dockerfile included in the repo.

# Simple examples

Make sure to bring up a CUDA container based on the docker file included here. The Rust call signatures for the channelizer are self documented by usage in the folder [``channelizer`` crate](channelizer/examples/revert_example.rs). The linked file shows by example how both the forward and revert work. An additional example usage maybe found in the [crate root](channelizer/src/lib.rs) under ``correctness_visual_test``.

Tests can be run using ``cargo test``. One can visualize the output channogram by simply running the ``pytest_lib_visual.py`` script. To visualize the result of revert, run ``pytest_revert.py`` to generate the png files of the spectrum.

# Milestone delivery

For this milestone, we want to show that our polyphase filter bank achieves greater than 30dB improvement in dynamic range. To test this, the user should follow these steps:

1. Bring this container up.

2. Inside the container, run 

```
cargo run --release --example three_tone_ota_test -- run \
    --tx_antenna <TX_ANTENNA> \
    --tx_channel <TX_CHANNEL> \
    --tx_addr <TX_ADDRESS> \
    --tx_usrp_type <TX_TYPE> \
    --tx_norm_gain <TX_NORMALIZED_GAIN> \
    --rx_antenna <RX_ANTENNA> \
    --rx_channel <RX_CHANNEL> \
    --rx_addr <RX_ADDRESS> \
    --rx_usrp_type <RX_TYPE> \
    --rx_norm_gain <RX_NORMALIZED_GAIN>
```

We found the following parameters to work well for us:
```
cargo run --release --example three_tone_ota_test -- run \
    --tx_antenna TX/RX \
    --tx_channel 0 \
    --tx_addr 192.168.101.20 \
    --tx_usrp_type x300 \
    --tx_norm_gain 0.2 \
    --rx_antenna TX/RX \
    --rx_channel 0 \
    --rx_addr 192.168.101.16 \
    --rx_usrp_type n3xx \
    --rx_norm_gain 0.6
```

This command instructs cargo to run the example [rust code here](channelizer/examples/three_tone_ota_test.rs). We are concerned with the three tone test scenario, where we have three narrow band tones, and one of them is **35dB** weaker than the other two. The rust file performs the following steps:

1. creates iq for the three tone test scenario.
2. transmit the iq over the air.
3. receive the iq over the air.
4. perform the channelization on both synthetic and ota data
5. save the [synthetic iq tones](iq/synthetic_tones.32cf), [ota iq tones](iq/ota_tones.32cf), [synthetic tones passed through channogram](iq/synthetic_tones_channelized.32cf), and [ota tones passed through channogram](iq/ota_tones_channelized.32cf) to disk.

3. Finally, to visualize the channogram, channpsd and their superior performance compared to spectrogram and psd, run the python script ``tone_test_spectrogram.py --multiplier <FACTOR>`` in [here](test_scripts). This will generate a ``png`` and ``txt`` files in the ``images`` folder. 

    1. **chann_spectrogram.png** 
    2. **fcs.txt**

This image will show the channogram (top plot) and the spectrogram (bottom plot) of the three tone test scenario.
The txt file will contain the center frequencies detected by the channogram and spectrogram.

> [!NOTE]
> The `--multiplier <FACTOR>` argument is optional. If provided, threshold for channogram is set to `FACTOR` times estimated noise floor. If not provided, threshold is set to 1.5 times estimated noise floor.