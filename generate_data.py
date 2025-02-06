from synthetic_data import *
import json
from parameter_arrays import *
from random_velocity_blob_factory import *

# Signal length


def gen_signal_length():
    full_outputs = [
        [
            run_sims(RunParameters(vxs[i], vys[i], T=T_case1[t], K=K_case1[t]))
            for i in np.arange(0, len(vxs))
        ]
        for t in np.arange(0, len(T_case1))
    ]
    full_outputs_Ly = [
        [
            run_sims(
                RunParameters(vxs[0], vys[0], T=T_case_Ly[t], K=K_case_Ly[t], Ly=100)
            )
        ]
        for t in np.arange(0, len(T_case_Ly))
    ]

    with open("data/full_outputs1.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)

    with open("data/full_outputs1_Ly.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs_Ly), file)


# Number of pulses


def gen_number_pulses():
    full_outputs = [
        [
            run_sims(RunParameters(vxs[i], vys[i], K=K_case2[t]))
            for i in np.arange(0, len(vxs))
        ]
        for t in np.arange(0, len(K_case2))
    ]
    full_outputs_Ly = [
        [run_sims(RunParameters(vxs[0], vys[0], K=K_case2[t], Ly=100))]
        for t in np.arange(0, len(K_case2))
    ]

    with open("data/full_outputs2.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)

    with open("data/full_outputs2_Ly.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs_Ly), file)


# Spatial resolution


def gen_spatial_resolution():
    full_outputs = [
        [
            run_sims(RunParameters(vxs[i], vys[i], delta=d))
            for i in np.arange(0, len(vxs))
        ]
        for d in deltas3
    ]
    full_outputs_dt = [
        [run_sims(RunParameters(vxs[0], vys[0], delta=d, dt=0.001))] for d in deltas3
    ]

    with open("data/full_outputs3.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)

    with open("data/full_outputs3_dt.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs_dt), file)


# Random velocities


def gen_random_velocities():
    full_outputs = [
        [
            run_sims(RunParameters(vxs[i], vys[i], sigma=s))
            for i in np.arange(0, len(vxs))
        ]
        for s in sigmas4
    ]

    with open("data/full_outputs4.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)


# Noise


def gen_noise():
    full_outputs = [
        [
            run_sims(RunParameters(vxs[i], vys[i], snr=1 / s))
            for i in np.arange(0, len(vxs))
        ]
        for s in snr5
    ]

    with open("data/full_outputs5.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)

    full_outputs = [
        [
            run_sims(RunParameters(1, 0, T=T_case5[i], K=T_case5[i], snr=1 / s))
            for i in np.arange(0, len(T_case5))
        ]
        for s in snr5
    ]

    with open("data/full_outputs5_time.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)


# Temporal resolution


def gen_temporal_resolution():
    full_outputs = [
        [run_sims(RunParameters(vxs[i], vys[i], dt=dt)) for i in np.arange(0, len(vxs))]
        for dt in delta_t6
    ]

    with open("data/full_outputs6.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)


# Other pulse shapes


def gen_other_pulse_shapes():
    full_outputs = [
        [
            run_sims(RunParameters(1, 0, bs_perp=bs, bs_prop=bs, delta=d))
            for bs in pulse_shapes7
        ]
        for d in deltas7
    ]

    with open("data/full_outputs7_delta.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)


# Elongated blobs


def gen_elongated_blobs():
    full_outputs = [
        [
            run_sims(RunParameters(vxs[i], vys[i], wx=wxs8[j], wy=1 / wxs8[j]))
            for i in np.arange(0, len(vxs))
        ]
        for j in np.arange(0, len(wxs8))
    ]
    full_outputs_delta = [
        [run_sims(RunParameters(vxs[0], vys[0], wx=wxs8[j], wy=1 / wxs8[j], delta=0.2))]
        for j in np.arange(0, len(wxs8))
    ]

    with open("data/full_outputs8.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)

    with open("data/full_outputs8_delta.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs_delta), file)


# Tilting


def gen_tilting():
    full_outputs = [
        [
            run_sims(RunParameters(vxs[0], vys[0], wx=wxs9[j], wy=1 / wxs9[j], theta=t))
            for j in np.arange(0, len(wxs9))
        ]
        for t in tilts9
    ]
    full_outputs_delta = [
        [
            run_sims(
                RunParameters(
                    vxs[0], vys[0], wx=wxs9[j], wy=1 / wxs9[j], theta=t, delta=0.2
                )
            )
            for j in np.arange(0, len(wxs9))
        ]
        for t in tilts9
    ]

    with open("data/full_outputs9.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)

    with open("data/full_outputs9_delta.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs_delta), file)


# Linear damping


def gen_linear_damping():
    full_outputs = [
        [
            run_sims(RunParameters(vxs[i], vys[i], taup=taups10[t]))
            for i in np.arange(0, len(vxs))
        ]
        for t in np.arange(0, len(taups10))
    ]

    with open("data/full_outputs10.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)


# Correlated amplitudes and velocities


def make_2d_realization_corr(rp: RunParameters):
    p0x, p0y = 1, 5
    xpoints = np.array([p0x, p0x + rp.delta])
    ypoints = np.array([p0y, p0y + rp.delta])
    if rp.sigma is not None:
        bf = CorrelatedBlobFactory(
            A_dist=DistributionEnum.deg,
            wx_dist=DistributionEnum.deg,
            vx_parameter=rp.vx,
            vy_parameter=rp.vy,
            wx_parameter=rp.wx,
            wy_parameter=rp.wy,
            blob_alignment=rp.blob_alignment,
            s=rp.sigma,
        )
    else:
        bf = DefaultBlobFactory(
            A_dist=DistributionEnum.deg,
            wx_dist=DistributionEnum.deg,
            vx_dist=DistributionEnum.deg,
            vy_dist=DistributionEnum.deg,
            vy_parameter=rp.vy,
            vx_parameter=rp.vx,
            wx_parameter=rp.wx,
            wy_parameter=rp.wy,
            blob_alignment=rp.blob_alignment,
        )
    if rp.theta != 0:
        bf.set_theta_setter(lambda: rp.theta)
    bm = Model(
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
    )
    update_geometry(xpoints, ypoints, bm)
    return bm.make_realization(speed_up=True, error=1e-10)


def run_params_corr(rp: RunParameters):
    ds = make_2d_realization_corr(rp)
    if rp.snr is None:
        ds = ve.SyntheticBlobImagingDataInterface(ds)
    else:
        ds = NoisyImagingDataInterface(ds, rp.snr)
    pd = ve.estimate_velocities_for_pixel(0, 0, ds, eo)
    return RunOutput(pd.vx, pd.vy, pd.confidence)


def run_sims_corr(rp: RunParameters):
    run_outputs = np.array([run_params_corr(rp) for i in range(sims_per_case)])
    return RunResults(rp, run_outputs)


def gen_corr():
    full_outputs = [
        [
            run_sims_corr(RunParameters(vxs[i], vys[i], sigma=s))
            for i in np.arange(0, len(vxs))
        ]
        for s in ss11
    ]

    with open("data/full_outputs11.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)


# Random sizes


def make_2d_realization_random(rp: RunParameters):
    p0x, p0y = 1, 5
    xpoints = np.array([p0x, p0x + rp.delta])
    ypoints = np.array([p0y, p0y + rp.delta])
    if rp.sigma is not None:
        bf = RandomBlobFactory(
            A_dist="deg",
            wx_dist="deg",
            vx_dist="deg",
            vy_dist="deg",
            vx_parameter=rp.vx,
            vy_parameter=rp.vy,
            wx_parameter=rp.wx,
            wy_parameter=rp.wy,
            blob_alignment=rp.blob_alignment,
            s=rp.sigma,
        )
    else:
        bf = DefaultBlobFactory(
            A_dist="deg",
            wx_dist="deg",
            vx_dist="deg",
            vy_dist="deg",
            vy_parameter=rp.vy,
            vx_parameter=rp.vx,
            wx_parameter=rp.wx,
            wy_parameter=rp.wy,
            blob_alignment=rp.blob_alignment,
        )
    if rp.theta != 0:
        bf.set_theta_setter(lambda: rp.theta)
    bm = Model(
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
    )
    update_geometry(xpoints, ypoints, bm)
    return bm.make_realization(speed_up=True, error=1e-10)


def run_params_random(rp: RunParameters):
    ds = make_2d_realization_random(rp)
    if rp.snr is None:
        ds = ve.SyntheticBlobImagingDataInterface(ds)
    else:
        ds = NoisyImagingDataInterface(ds, rp.snr)
    pd = ve.estimate_velocities_for_pixel(0, 0, ds, eo)
    return RunOutput(pd.vx, pd.vy, pd.confidence)


def run_sims_random(rp: RunParameters):
    run_outputs = np.array([run_params_random(rp) for i in range(sims_per_case)])
    return RunResults(rp, run_outputs)


def gen_random_size():
    full_outputs = [
        [
            run_sims_random(RunParameters(vxs[i], vys[i], sigma=s))
            for i in np.arange(0, len(vxs))
        ]
        for s in ss12
    ]

    with open("data/full_outputs12.json", "w") as file:
        json.dump(serialize_full_outputs(full_outputs), file)


# gen_signal_length()
# gen_number_pulses()
# gen_spatial_resolution()
# gen_random_velocities()
# gen_noise()
# gen_temporal_resolution()
# gen_other_pulse_shapes()
# gen_elongated_blobs()
gen_tilting()
# gen_linear_damping()
# gen_corr()
# gen_random_size()
