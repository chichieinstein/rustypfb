This is an implementation of a Non-maximally decimated Polyphase Filter Bank (PFB) which is callable from the Rust programming Language. This filter bank supports perfect reconstruction. The backend is written in CUDA C++ for real time deployment. The image below illustrates the frequency responses of both the analysis and synthesis prototype filters used in the filter bank. The analysis filter is a simple Nyquist filter, and the passband of the synthesis filter is designed to completely cover the transition band of the analysis filter. This ensures distortion free reconstruction.

The forward channelizer process function has been benchmarked to attain a throughput of ``~763 Megasamples per second`` on a single NVIDIA A10 GPU core. In order to check things are running as required, the test written in the root of the ``channelizer`` crate that creates channelized output needs to be run. All dependencies are listed in the Dockerfile included in the repo.

![Image Alt Text](/docs/filter_responses.png)

For alias free reconstruction, we need minimal overlap between the passbands of the untranslated synthesis filter and the analysis filter when it is shifted by the downsampling factor. 

![Image Alt Text](/docs/shifted_filter_responses.png)

This concludes the theoretical justification for the choices of prototype synthesis and analysis filters. More details on how to arrive at these conditions maybe found in this [paper](https://ieeexplore.ieee.org/document/6690219).

Cosider the following figure for the dataset where many energies are crowded together to create a busy spectrum (we call this the LPI combined example).
![Image Alt Text](/docs/LPI.png)

Depicted here, is the comparison of the reverted spectrum (i.e., the stft of the reverted IQ) with the channogram obtained by forward channelization by the PFB. The prototype analysis and synthesis filters are chosen as above. 

Here is the same comparison for the case where three different DSSS transmissions occur in the spectrum.
![Image Alt Text](/docs/DSSS.png)

In both the cases, we see no aliasing artifacts, and no distortions in the reverted spectrum. By definition, this is perfect reconstruction!

# How to use the channelizer via examples
The Rust call signatures for the channelizer are self-documented by usage in the folder [``channelizer`` crate](https://github.com/ucsdwcsng/rustypfb/tree/main/channelizer/examples). The linked folder contains the ``revert_example.rs`` file, which shows by example how both the forward and revert work. An additional example usage may be found in the [crate root](https://github.com/ucsdwcsng/rustypfb/blob/main/channelizer/src/lib.rs) under ``correctness_visual_test``. One can visualize the output channogram by simply running the ``pytest_lib_visual.py`` script. To visualize the result of revert, run ``pytest_revert.py`` to generate the png examples linked here.


