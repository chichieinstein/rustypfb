This is an implementation of a non-maximally decimating polyphase filter bank (PFB), which supports perfect reconstruction. The filter bank has been written to be usable as a Rust crate. The backend is written in CUDA C++ for real time deployment.

In order to check things are running as required, the test written in the root of the ``channelizer`` crate needs to be run. All dependencies are listed in the Dockerfile included in the repo.

# Simple examples

Make sure to bring up a CUDA container based on the docker file included here. The Rust call signatures for the channelizer are self documented by usage in the folder [``channelizer`` crate](channelizer/examples/revert_example.rs). The linked file shows by example how both the forward and revert work. An additional example usage maybe found in the [crate root](channelizer/src/lib.rs) under ``correctness_visual_test``.

Tests can be run using ``cargo test``. One can visualize the output channogram by simply running the ``pytest_lib_visual.py`` script. To visualize the result of revert, run ``pytest_revert.py`` to generate the png files of the spectrum.

# Milestone delivery

For this milestone, we want to show that our polyphase filter bank achieves greater than 30dB improvement in dynamic range. To test this, the user should follow these steps:

1. Bring this container up.

2. Inside the container, run 

``cargo run --example three_tone_test`` 

This command instructs cargo to run the example [rust code here](channelizer/examples/three_tone_test.rs). This performs two tasks, namely,

    1. creates iq for the following test scenario: three narrow band tones are generated, and one of them is **35dB** weaker than the other two. 
    2. Processes the iq generated in the previous step with the polyphase filter bank over 1024 channels and stores the channelized iq.

3. Finally, to visualize the channogram, channpsd and their superior performance compared to spectrogram and psd, run the python script ``tone_test_spectrogram.py`` in [here](test_scripts). This will generate two ``png`` files, namely

    1. **channogram.png** 
    2. **spectrogram.png** 

in a folder named ``images``. These images will illustrate the achieved improved dynamic range with the channelizer.



