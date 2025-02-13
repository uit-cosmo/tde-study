from blobmodel import (
    Model,
    DefaultBlobFactory,
    BlobShapeImpl,
    BlobFactory,
    Blob,
    AbstractBlobShape,
    DistributionEnum,
)
import matplotlib as mpl
import cosmoplots
from estimation import *
from random_velocity_blob_factory import RandomVelocityBlobFactory
import velocity_estimation as ve


eo = ve.EstimationOptions()
eo.method = ve.TDEMethod.CC
eo.neighbour_options.ccf_min_lag = -1
eo.cc_options.minimum_cc_value = 0
eo.cc_options.interpolate = True
eo.cc_options.running_mean = False
eo.cc_options.cc_window = 1000

sims_per_case = 10


def update_geometry(x_grid, y_grid, model):
    x_matrix, y_matrix, t_matrix = np.meshgrid(x_grid, y_grid, model._geometry.t)
    model._geometry.x_matrix = x_matrix
    model._geometry.y_matrix = y_matrix
    model._geometry.t_matrix = t_matrix
    model._geometry.Ny = len(y_grid)
    model._geometry.Nx = len(x_grid)
    model._geometry.x = x_grid
    model._geometry.y = y_grid


def make_2d_realization(rp: RunParameters):
    p0x, p0y = 1, 5
    xpoints = np.array([p0x, p0x + rp.delta])
    ypoints = np.array([p0y, p0y + rp.delta])
    if rp.sigma is not None:
        bf = RandomVelocityBlobFactory(
            vx_parameter=rp.vx,
            vy_parameter=rp.vy,
            wx_parameter=rp.wx,
            wy_parameter=rp.wy,
            blob_alignment=rp.blob_alignment,
            sigma=rp.sigma,
        )
    else:
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
        Nx=10,
        Ny=1,
        Lx=rp.Lx,
        Ly=rp.Ly,
        dt=rp.dt,
        T=rp.T,
        num_blobs=rp.K,
        blob_shape=BlobShapeImpl(rp.bs_prop, rp.bs_perp),
        periodic_y=False,
        t_drain=rp.taup,
        blob_factory=bf,
        verbose=False,
        t_init=10
    )
    update_geometry(xpoints, ypoints, model)
    return model.make_realization(speed_up=True, error=1e-10)


def make_2d_realization_full_resolution(rp: RunParameters):
    p0x, p0y = 1, 5
    if rp.sigma is not None:
        bf = RandomVelocityBlobFactory(
            vx_parameter=rp.vx,
            vy_parameter=rp.vy,
            wx_parameter=rp.wx,
            wy_parameter=rp.wy,
            blob_alignment=rp.blob_alignment,
            sigma=rp.sigma,
        )
    else:
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
        Nx=int(rp.Lx/rp.delta),
        Ny=int(rp.Ly/rp.delta),
        Lx=rp.Lx,
        Ly=rp.Ly,
        dt=rp.dt,
        T=rp.T,
        num_blobs=rp.K,
        blob_shape=BlobShapeImpl(rp.bs_prop, rp.bs_perp),
        periodic_y=False,
        t_drain=rp.taup,
        blob_factory=bf,
        verbose=False,
        t_init=10
    )
    return model.make_realization(speed_up=True, error=1e-10)


def run_params(rp: RunParameters):
    ds = make_2d_realization(rp)
    if rp.snr is None:
        ds = ve.SyntheticBlobImagingDataInterface(ds)
    else:
        ds = NoisyImagingDataInterface(ds, rp.snr)
    pd = ve.estimate_velocities_for_pixel(0, 0, ds, eo)
    return RunOutput(pd.vx, pd.vy, pd.confidence)


def run_sims(rp: RunParameters):
    run_outputs = np.array([run_params(rp) for i in range(sims_per_case)])
    return RunResults(rp, run_outputs)


def serialize_full_outputs(full_outputs):
    return [[result.to_dict() for result in sublist] for sublist in full_outputs]


def deserialize_full_outputs(serialized_full_outputs):
    return [
        [RunResults.from_dict(result_dict) for result_dict in sublist]
        for sublist in serialized_full_outputs
    ]


def get_mse_for_slice(full_outputs, i):
    deviation_vxs = np.array(
        [
            [c.out_vx - d[i].run_params.vx for c in d[i].run_outputs]
            for d in full_outputs
        ]
    )
    deviation_vys = np.array(
        [
            [c.out_vy - d[i].run_params.vy for c in d[i].run_outputs]
            for d in full_outputs
        ]
    )
    return ((deviation_vxs) ** 2 + (deviation_vys) ** 2).mean(axis=1)


def get_max_tau_for_slice(full_outputs, i):
    out_vxs = np.array([[c.out_vx for c in d[i].run_outputs] for d in full_outputs])
    out_vys = np.array([[c.out_vy for c in d[i].run_outputs] for d in full_outputs])
    deltas = np.array([d[i].run_params.delta for d in full_outputs])
    u2 = out_vxs**2 + out_vys**2
    taux = np.abs(deltas[:, np.newaxis] * out_vxs / u2)
    tauy = np.abs(deltas[:, np.newaxis] * out_vys / u2)
    return np.maximum(taux, tauy).mean(axis=1)


def get_std_for_slice(full_outputs, i):
    out_vxs = np.array([[c.out_vx for c in d[i].run_outputs] for d in full_outputs])
    out_vys = np.array([[c.out_vy for c in d[i].run_outputs] for d in full_outputs])
    return np.sqrt(out_vxs.std(axis=1) ** 2 + out_vys.std(axis=1) ** 2)


def get_velocities(full_outputs, i):
    out_vxs = np.array([[c.out_vx for c in d[i].run_outputs] for d in full_outputs])
    out_vys = np.array([[c.out_vy for c in d[i].run_outputs] for d in full_outputs])
    return out_vxs, out_vys


def get_confidence_for_slice(full_outputs, i):
    confidences = np.array(
        [[c.confidence for c in d[i].run_outputs] for d in full_outputs]
    )
    return confidences.mean(axis=1)


def plot_func(ax, x, full_outputs, func=get_mse_for_slice, mode=0, threshold_line=True):
    if mode == 0:
        ax.scatter(x, func(full_outputs, 0), color="blue")
        ax.scatter(x, func(full_outputs, 1), color="red")
        ax.scatter(x, func(full_outputs, 2), color="green")
        if threshold_line:
            ax.hlines(0.1, 0, np.max(x), color="black", ls="--", lw=0.5)

        ax.set_ylim(1e-5, 10)
        ax.set_yscale("log")
        ax.set_xscale("log")

    if mode == 1:
        ax.scatter(x, func(full_outputs, 0), color="blue")
        ax.scatter(x, func(full_outputs, 1), color="red")
        ax.scatter(x, func(full_outputs, 2), color="green")
        if threshold_line:
            ax.hlines(0.1, 0, np.max(x), color="black", ls="--", lw=0.5)

        ax.set_ylim(-0.1, 2)
        ax.set_xscale("log")


def plot_ccf(ax, x, errors, xlabel):
    ax.scatter(x, get_confidence_for_slice(full_outputs, 0), color="blue")
    ax.scatter(x, get_confidence_for_slice(full_outputs, 1), color="red")
    ax.scatter(x, get_confidence_for_slice(full_outputs, 2), color="green")

    ax.set_ylim(0, 1.1)
    ax.set_xscale("log")
    ax.set_ylabel("$\text{C}$")
    ax.set_xlabel(xlabel)


MSE = r"$\sigma_\text{MSE}$"
hatch = "/////"
