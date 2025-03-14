import matplotlib.pyplot as plt

from synthetic_data import *
import blobmodel as bm
import json
from random_velocity_blob_factory import *
from show_model import *

rp = RunParameters(
    vx=1,
    vy=1,
    T=20,
    K=10,
    dt=0.1,
    delta=0.5,
    bs_perp=bm.BlobShapeEnum.gaussian,
    bs_prop=bm.BlobShapeEnum.gaussian,
)

# ds = make_2d_realization(rp)
ds = make_2d_realization_full_resolution(rp)

data = ds.isel(x=0, y=0)["n"].values
time = ds.isel(x=0, y=0)["t"].values
show_model(ds)
