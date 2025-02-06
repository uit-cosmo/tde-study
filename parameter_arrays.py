import numpy as np
import blobmodel as bm

velocity_dirs = np.array([0, np.pi / 6, -np.pi / 3])
vxs = np.cos(velocity_dirs)
vys = np.sin(velocity_dirs)

# Signal length

T_case1 = np.array([int(i) for i in np.logspace(1, 4, num=20)])
K_case1 = T_case1

T_case_Ly = T_case1[T_case1 > 100]
K_case_Ly = T_case_Ly

# Number of pulses

K_case2 = np.array([int(i) for i in np.logspace(0, 4, num=20)])

# Spatial resolution

deltas3 = np.logspace(np.log10(0.001), np.log10(10), num=20)

# Random velocities

sigmas4 = np.logspace(np.log10(0.01), 1, num=20)

# Noise

snr5 = np.logspace(np.log10(0.001), 1, num=20)
T_case5 = np.array([100, 1000, 10000])

# Temporal resolution

delta_t6 = np.logspace(np.log10(0.001), 1, num=20)

# Other pulse shapes

pulse_shapes7 = [
    bm.BlobShapeEnum.gaussian,
    bm.BlobShapeEnum.exp,
    bm.BlobShapeEnum.double_exp,
    bm.BlobShapeEnum.lorentz,
    bm.BlobShapeEnum.secant,
    bm.BlobShapeEnum.rect,
]
blob_direction7 = np.arange(-np.pi / 2, np.pi / 2, step=np.pi / 20)
deltas7 = np.logspace(np.log10(0.001), np.log10(10), num=20)

# Elongated blobs

aspect_ratios8 = np.logspace(-1, 1, num=20)
wxs8 = np.sqrt(aspect_ratios8)

# Tilting

tilts9 = np.arange(-np.pi / 2, np.pi / 2, step=np.pi / 50)
aspect_ratios9 = np.array([1 / 4, 4])
wxs9 = np.sqrt(aspect_ratios9)

# Linear damping

taups10 = np.logspace(-0.5, 2, num=50)

# Correlated amplitudes and velocities

ss11 = np.arange(0, 2, 0.1)

# Random sizes

ss12 = np.arange(0, 1.99, 0.1)
