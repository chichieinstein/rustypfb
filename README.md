This is an implementation of a non-maximally decimating polyphase filter bank (PFB), which supports perfect reconstruction. The filter bank has been written to be usable as a Rust crate. The backend is written in CUDA C++ for real time deployment.

In order to check things are running as required, the test written in the root of the ``channelizer`` crate needs to be run. All dependencies are listed in the Dockerfile included in the repo.

# How to use the channelizer via examples

Make sure to bring up a CUDA container based on the docker file included here. The Rust call signatures for the channelizer are self documented by usage in the folder [``channelizer`` crate](channelizer/examples). The linked folder contains the ``revert_example.rs`` file, which shows by example how both the forward and revert work. An additional example usage maybe found in the [crate root](channelizer/src/lib.rs) under ``correctness_visual_test``.

Tests can be run using ``cargo test``. One can visualize the output channogram by simply running the ``pytest_lib_visual.py`` script. To visualize the result of revert, run ``pytest_revert.py`` to generate the png files of the spectrum. 
