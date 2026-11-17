# ReDMark MT-Net

Source: https://github.com/MahdiShAhmadi/ReDMark . Keep its license and attribution. Place the author checkout at WM_ROOT/third_party/redmark, including logs/MT-Net/Weights/weights_final.h5. Weights are not redistributed.

adapter.py exports ReDMark. It preserves the author's float64 NumPy patch/tiling boundary; the TensorFlow scalar input shape is adapted for TF2.15 without changing learned layers. Every learned tensor is checked against the HDF5 checkpoint. Native1024 information bits are placed in64 blocks with four-copy diagonal repetition. Useful payload is the first32 bits; A pads992 zeros, B uses independent seeded random bits. Strict majority ties become zero. Default alpha1. RGB embedding modifies Y while preserving chroma; received images resize to512. This is a disclosed RGB adaptation of a grayscale benchmark.

Dependencies: TensorFlow2.15.1, NumPy1.26.4, SciPy, h5py, OpenCV and the project PyTorch stack. Set WM_ROOT before importing. Use ReDMark().embed(rgb_uint8,bits1024) and .extract(received_rgb); the latter returns scores and threshold0.5. See [shared protocol and results](../team10/README.md).
