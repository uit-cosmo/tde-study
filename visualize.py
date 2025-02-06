from synthetic_data import *
import json
from parameter_arrays import *
from random_velocity_blob_factory import *
from blobmodel import show_model, BlobShapeEnum


def make_visualization_realization(rp: RunParameters):
    bf = DefaultBlobFactory(
        A_dist=DistributionEnum.deg,
        vy_parameter=rp.vy,
        vx_parameter=rp.vx,
        wx_parameter=rp.wx,
        wy_parameter=rp.wy,
        blob_alignment=rp.blob_alignment,
    )
    if rp.theta != 0:
        bf.set_theta_setter(lambda: rp.theta)
    model = Model(
        Nx=32,
        Ny=32,
        Lx=rp.Lx,
        Ly=rp.Ly,
        dt=rp.dt,
        T=rp.T,
        num_blobs=rp.K,
        blob_shape=BlobShapeImpl(rp.bs_prop, rp.bs_perp),
        periodic_y=False,
        t_drain=rp.taup,
        blob_factory=bf,
        t_init=10,
        verbose=False,
    )
    return model.make_realization(speed_up=True, error=1e-10)


rp = RunParameters(vxs[0], vys[0], wx=2, wy=1 / 2, theta=np.pi / 4)
rp.T = 20
rp.dt = 0.1
rp.bs_perp = BlobShapeEnum.rect
rp.bs_prop = BlobShapeEnum.rect
rp.K = 10
ds = make_visualization_realization(rp)
show_model(ds)
