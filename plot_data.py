from synthetic_data import *
from utils import *
import json
import matplotlib.pyplot as plt
import cosmoplots as cp
from parameter_arrays import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
    r"\newcommand{\taup}{\ensuremath{\tau_\shortparallel}} \newcommand{\wh}{\widehat}"
    r" \newcommand{\ave}[1]{{\left<#1\right>}}"
)

fig, ax = cp.figure_multiple_rows_columns(1, 2)

with open("data/full_outputs1.json", "r") as file:
    full_outputs = deserialize_full_outputs(json.load(file))

with open("data/full_outputs1_Ly.json", "r") as file:
    full_outputs_Ly = deserialize_full_outputs(json.load(file))

print(
    "Plotting average over {} simulations".format(len(full_outputs[0][0].run_outputs))
)

dt = full_outputs[0][0].run_params.dt
average_max_tau = (
    get_max_tau_for_slice(full_outputs, 0)
    + get_max_tau_for_slice(full_outputs, 1)
    + get_max_tau_for_slice(full_outputs, 2)
) / 3
ax[0].fill_between(T_case1[average_max_tau < dt], -0.5, 1.1, color="red", alpha=0.5)
ax[1].fill_between(T_case1[average_max_tau < dt], -0.5, 1.1, color="red", alpha=0.5)

plot_func(ax[0], T_case1, full_outputs, func=get_mse_for_slice, mode=1)
plot_func(
    ax[1], T_case1, full_outputs, func=get_std_for_slice, mode=1, threshold_line=False
)

ax[0].scatter(
    T_case_Ly, get_mse_for_slice(full_outputs_Ly, 0), color="blue", marker="v"
)
ax[1].scatter(
    T_case_Ly, get_std_for_slice(full_outputs_Ly, 0), color="blue", marker="v"
)

ax[0].set_ylabel(r"$\text{MSE}$")
ax[1].set_ylabel(MSE)
ax[0].set_xlabel(r"$T/\tau_\text{d}$")
ax[1].set_xlabel(r"$T/\tau_\text{d}$")
ax[0].set_ylim(-0.1, 1)
ax[1].set_ylim(-0.1, 1)

plt.savefig("method_study_T.eps", bbox_inches="tight")

# Number of pulses

fig, ax = cp.figure_multiple_rows_columns(1, 2)

with open("data/full_outputs2.json", "r") as file:
    full_outputs = deserialize_full_outputs(json.load(file))

with open("data/full_outputs2_Ly.json", "r") as file:
    full_outputs_Ly = deserialize_full_outputs(json.load(file))

print(
    "Plotting average over {} simulations".format(len(full_outputs[0][0].run_outputs))
)

plot_func(ax[0], K_case2, full_outputs, func=get_mse_for_slice, mode=1)
plot_func(
    ax[1], K_case2, full_outputs, func=get_std_for_slice, mode=1, threshold_line=False
)

ax[0].scatter(K_case2, get_mse_for_slice(full_outputs_Ly, 0), color="blue", marker="v")
ax[1].scatter(K_case2, get_std_for_slice(full_outputs_Ly, 0), color="blue", marker="v")

average_confidence = (
    get_confidence_for_slice(full_outputs, 0)
    + get_confidence_for_slice(full_outputs, 1)
    + get_confidence_for_slice(full_outputs, 2)
) / 3
ax[0].fill_between(
    K_case2[average_confidence < 0.25],
    -0.5,
    1.1,
    color="lightgray",
    alpha=0.5,
    edgecolor="none",
)
ax[1].fill_between(
    K_case2[average_confidence < 0.25],
    -0.5,
    1.1,
    color="lightgray",
    alpha=0.5,
    edgecolor="none",
)

ax[0].set_ylabel(r"$\text{MSE}$")
ax[1].set_ylabel(MSE)
ax[0].set_xlabel(r"$K$")
ax[1].set_xlabel(r"$K$")
ax[0].set_ylim(-0.1, 1)
ax[1].set_ylim(-0.1, 1)

plt.savefig("method_study_K.eps", bbox_inches="tight")

# Spatial resolution

fig, ax = cp.figure_multiple_rows_columns(1, 2)


with open("data/full_outputs3.json", "r") as file:
    full_outputs = deserialize_full_outputs(json.load(file))

with open("data/full_outputs3_dt.json", "r") as file:
    full_outputs_dt = deserialize_full_outputs(json.load(file))

print(
    "Plotting average over {} simulations".format(len(full_outputs[0][0].run_outputs))
)

average_confidence = (
    get_confidence_for_slice(full_outputs, 0)
    + get_confidence_for_slice(full_outputs, 1)
    + get_confidence_for_slice(full_outputs, 2)
) / 3
grey_delta_start = deltas3[np.argmax([average_confidence < 0.25])]
ax[0].fill_between(
    [grey_delta_start, 100], -0.5, 1.1, color="lightgray", alpha=0.5, edgecolor="none"
)
ax[1].fill_between(
    deltas3[average_confidence < 0.25],
    -0.5,
    1.1,
    color="lightgray",
    alpha=0.5,
    edgecolor="none",
)

average_max_tau = (
    get_max_tau_for_slice(full_outputs, 0)
    + get_max_tau_for_slice(full_outputs, 1)
    + get_max_tau_for_slice(full_outputs, 2)
) / 3
ax[0].fill_between(
    deltas3[average_max_tau < dt],
    -0.5,
    1.1,
    color="none",
    edgecolor="lightgrey",
    hatch=hatch,
    linewidth=0.0,
)
ax[1].fill_between(
    deltas3[average_max_tau < dt],
    -0.5,
    1.1,
    color="none",
    edgecolor="lightgrey",
    hatch=hatch,
    linewidth=0.0,
)

plot_func(ax[0], deltas3, full_outputs, func=get_mse_for_slice, mode=1)
plot_func(
    ax[1], deltas3, full_outputs, func=get_std_for_slice, mode=1, threshold_line=False
)

ax[0].set_ylabel(r"$\text{MSE}$")
ax[1].set_ylabel(MSE)
ax[0].set_xlabel(r"$\Delta/\ell$")
ax[1].set_xlabel(r"$\Delta/\ell$")
ax[0].set_ylim(-0.1, 1)
ax[1].set_ylim(-0.1, 1)
ax[0].set_xlim(1e-3, 10)
ax[1].set_xlim(1e-3, 10)

plt.savefig("method_study_delta.eps", bbox_inches="tight")

# Random velocities

fig, ax = cp.figure_multiple_rows_columns(1, 2)

with open("data/full_outputs4.json", "r") as file:
    full_outputs = deserialize_full_outputs(json.load(file))

print(
    "Plotting average over {} simulations".format(len(full_outputs[0][0].run_outputs))
)

average_confidence = (
    get_confidence_for_slice(full_outputs, 0)
    + get_confidence_for_slice(full_outputs, 1)
    + get_confidence_for_slice(full_outputs, 2)
) / 3
ax[0].fill_between(
    sigmas4[average_confidence < 0.25], -0.5, 1.1, color="lightgray", alpha=0.5
)
ax[1].fill_between(
    sigmas4[average_confidence < 0.25], -0.5, 1.1, color="lightgray", alpha=0.5
)

plot_func(ax[0], sigmas4, full_outputs, func=get_mse_for_slice, mode=1)
plot_func(
    ax[1], sigmas4, full_outputs, func=get_std_for_slice, mode=1, threshold_line=False
)

ax[0].set_ylabel(r"$\text{MSE}$")
ax[1].set_ylabel(MSE)
ax[0].set_xlabel(r"$\sigma$")
ax[1].set_xlabel(r"$\sigma$")
ax[0].set_ylim(-0.1, 1)
ax[1].set_ylim(-0.1, 1)

plt.savefig("method_study_rand_v.eps", bbox_inches="tight")

# Noise

fig, ax = cp.figure_multiple_rows_columns(1, 2)

with open("data/full_outputs5_time.json", "r") as file:
    full_outputs = deserialize_full_outputs(json.load(file))

print(
    "Plotting average over {} simulations".format(len(full_outputs[0][0].run_outputs))
)

average_confidence = (
    get_confidence_for_slice(full_outputs, 0)
    + get_confidence_for_slice(full_outputs, 1)
    + get_confidence_for_slice(full_outputs, 2)
) / 3
index_cc_fail = np.argmax(average_confidence < 0.25)
ax[0].fill_between(
    snr5[average_confidence < 0.25], -0.5, 1.1, color="lightgray", alpha=0.5
)
ax[1].fill_between(
    snr5[average_confidence < 0.25], -0.5, 1.1, color="lightgray", alpha=0.5
)

ax[0].scatter(
    snr5,
    get_mse_for_slice(full_outputs, 0),
    color="blue",
    label=r"$T/\tau_\text{d}=10^2$",
)
ax[0].scatter(
    snr5,
    get_mse_for_slice(full_outputs, 1),
    color="red",
    label=r"$T/\tau_\text{d}=10^3$",
)
ax[0].scatter(
    snr5,
    get_mse_for_slice(full_outputs, 2),
    color="green",
    label=r"$T/\tau_\text{d}=10^4$",
)
ax[0].hlines(0.1, 0, np.max(snr5), color="black", ls="--", lw=0.5)
ax[0].set_ylim(-0.1, 1)
ax[0].set_xscale("log")

ax[1].scatter(snr5, get_std_for_slice(full_outputs, 0), color="blue")
ax[1].scatter(snr5, get_std_for_slice(full_outputs, 1), color="red")
ax[1].scatter(snr5, get_std_for_slice(full_outputs, 2), color="green")
ax[1].set_ylim(-0.1, 1)
ax[1].set_xscale("log")

ax[0].set_ylabel(r"$\text{MSE}$")
ax[1].set_ylabel(MSE)
ax[0].set_xlabel(r"$\epsilon$")
ax[1].set_xlabel(r"$\epsilon$")
ax[0].set_ylim(-0.1, 1)
ax[1].set_ylim(-0.1, 1)

ax[0].set_xlim(1e-3, 10)
ax[1].set_xlim(1e-3, 10)
ax[0].legend()

plt.savefig("method_study_noise.eps", bbox_inches="tight")

# Temporal resolution

fig, ax = cp.figure_multiple_rows_columns(1, 2)

with open("data/full_outputs6.json", "r") as file:
    full_outputs = deserialize_full_outputs(json.load(file))

print(
    "Plotting average over {} simulations".format(len(full_outputs[0][0].run_outputs))
)

average_max_tau = (
    get_max_tau_for_slice(full_outputs, 0)
    + get_max_tau_for_slice(full_outputs, 1)
    + get_max_tau_for_slice(full_outputs, 2)
) / 3
ax[0].fill_between(
    delta_t6[average_max_tau < delta_t6],
    -0.5,
    1.1,
    color="none",
    edgecolor="lightgrey",
    hatch=hatch,
    linewidth=0.0,
)
ax[1].fill_between(
    delta_t6[average_max_tau < delta_t6],
    -0.5,
    1.1,
    color="none",
    edgecolor="lightgrey",
    hatch=hatch,
    linewidth=0.0,
)

plot_func(ax[0], delta_t6, full_outputs, func=get_mse_for_slice, mode=1)
plot_func(
    ax[1], delta_t6, full_outputs, func=get_std_for_slice, mode=1, threshold_line=False
)

ax[0].set_ylabel(r"$\text{MSE}$")
ax[1].set_ylabel(MSE)
xlabel = r"$\Delta t/\tau_\text{d}$"
ax[0].set_xlabel(xlabel)
ax[1].set_xlabel(xlabel)
ax[0].set_ylim(-0.1, 1)
ax[1].set_ylim(-0.1, 1)

ax[0].set_xlim(1e-3, 10)
ax[1].set_xlim(1e-3, 10)

plt.savefig("method_study_delta_t.eps", bbox_inches="tight")

# Other pulse shapes

fig, ax = cp.figure_multiple_rows_columns(1, 2)

with open("data/full_outputs7_delta.json", "r") as file:
    full_outputs = deserialize_full_outputs(json.load(file))

xlabel = r"$\Delta/\ell$"

average_confidence = (
    get_confidence_for_slice(full_outputs, 0)
    + get_confidence_for_slice(full_outputs, 1)
    + get_confidence_for_slice(full_outputs, 2)
    + get_confidence_for_slice(full_outputs, 3)
    + get_confidence_for_slice(full_outputs, 4)
    + get_confidence_for_slice(full_outputs, 5)
) / 6
average_confidence = np.nan_to_num(average_confidence)
ax[0].fill_between(
    deltas7[average_confidence < 0.25], -0.5, 1.1, color="lightgray", alpha=0.5
)
ax[1].fill_between(
    deltas7[average_confidence < 0.25], -0.5, 1.1, color="lightgray", alpha=0.5
)

average_max_tau = (
    get_max_tau_for_slice(full_outputs, 0)
    + get_max_tau_for_slice(full_outputs, 1)
    + get_max_tau_for_slice(full_outputs, 2)
    + get_max_tau_for_slice(full_outputs, 3)
    + get_max_tau_for_slice(full_outputs, 4)
    + get_max_tau_for_slice(full_outputs, 5)
) / 6
ax[0].fill_between(
    deltas7[average_max_tau < dt],
    -0.5,
    1.1,
    color="none",
    edgecolor="lightgrey",
    hatch=hatch,
    linewidth=0.0,
)
ax[1].fill_between(
    deltas7[average_max_tau < dt],
    -0.5,
    1.1,
    color="none",
    edgecolor="lightgrey",
    hatch=hatch,
    linewidth=0.0,
)

ax[0].scatter(deltas7, get_mse_for_slice(full_outputs, 0), color="blue", label="Gauss")
ax[0].scatter(deltas7, get_mse_for_slice(full_outputs, 1), color="red", label="Exp")
ax[0].scatter(deltas7, get_mse_for_slice(full_outputs, 2), color="green", label="2-Exp")
ax[0].scatter(
    deltas7,
    get_mse_for_slice(full_outputs, 3),
    color="blue",
    ls="--",
    label="Lor",
    marker="v",
)
ax[0].scatter(
    deltas7,
    get_mse_for_slice(full_outputs, 4),
    color="red",
    ls="--",
    label="Sec",
    marker="v",
)
ax[0].scatter(
    deltas7,
    get_mse_for_slice(full_outputs, 5),
    color="green",
    ls="--",
    label="Rec",
    marker="v",
)

ax[0].hlines(0.1, np.min(deltas7), np.max(deltas7), color="black", ls="--", lw=0.5)
ax[0].set_ylim(-0.1, 1.1)
ax[1].set_ylim(-0.1, 1.1)
ax[0].set_ylabel(r"$\text{MSE}$")
ax[1].set_ylabel(MSE)
ax[0].set_xlabel(xlabel)
ax[1].set_xlabel(xlabel)
ax[0].set_xscale("log")
ax[1].set_xscale("log")

ax[0].set_xlim([1e-3, 10])
ax[1].set_xlim([1e-3, 10])

ax[1].scatter(deltas7, get_std_for_slice(full_outputs, 0), color="blue", label="Gauss")
ax[1].scatter(deltas7, get_std_for_slice(full_outputs, 1), color="red", label="Exp")
ax[1].scatter(deltas7, get_std_for_slice(full_outputs, 2), color="green", label="2-Exp")
ax[1].scatter(
    deltas7,
    get_std_for_slice(full_outputs, 3),
    color="blue",
    ls="--",
    label="Lor",
    marker="v",
)
ax[1].scatter(
    deltas7,
    get_std_for_slice(full_outputs, 4),
    color="red",
    ls="--",
    label="Sec",
    marker="v",
)
ax[1].scatter(
    deltas7,
    get_std_for_slice(full_outputs, 5),
    color="green",
    ls="--",
    label="Rec",
    marker="v",
)

ax[1].legend(loc=9)

plt.savefig("method_study_pulse_shapes.eps", bbox_inches="tight")

# Elongated blobs

fig, ax = cp.figure_multiple_rows_columns(1, 2)

with open("data/full_outputs8.json", "r") as file:
    full_outputs = deserialize_full_outputs(json.load(file))

with open("data/full_outputs8_delta.json", "r") as file:
    full_outputs_delta = deserialize_full_outputs(json.load(file))

xlabel = r"$\ell_\shortparallel/\ell_\perp$"

average_confidence = (
    get_confidence_for_slice(full_outputs, 0)
    + get_confidence_for_slice(full_outputs, 1)
    + get_confidence_for_slice(full_outputs, 2)
) / 3
index_cc_fail = np.argmax(average_confidence < 0.25)
ax[0].fill_between(
    (wxs8**2)[average_confidence < 0.25], -0.5, 1.1, color="lightgray", alpha=0.5
)
ax[1].fill_between(
    (wxs8**2)[average_confidence < 0.25], -0.5, 1.1, color="lightgray", alpha=0.5
)

plot_func(ax[0], wxs8**2, full_outputs, func=get_mse_for_slice, mode=1)
plot_func(
    ax[1], wxs8**2, full_outputs, func=get_std_for_slice, mode=1, threshold_line=False
)

ax[0].scatter(
    wxs8**2, get_mse_for_slice(full_outputs_delta, 0), color="blue", marker="v"
)
ax[1].scatter(
    wxs8**2, get_std_for_slice(full_outputs_delta, 0), color="blue", marker="v"
)

ax[0].set_ylabel(r"$\text{MSE}$")
ax[1].set_ylabel(MSE)
ax[0].set_xlabel(xlabel)
ax[1].set_xlabel(xlabel)
ax[0].set_ylim(-0.1, 1)
ax[1].set_ylim(-0.1, 1)

ax[1].set_xlim(1e-1, 10)
ax[0].set_xlim(1e-1, 10)

plt.savefig("method_study_aspect_ratio_aligned.eps", bbox_inches="tight")

# Tilting

fig, ax = cp.figure_multiple_rows_columns(1, 2)

with open("data/full_outputs9.json", "r") as file:
    full_outputs = deserialize_full_outputs(json.load(file))

with open("data/full_outputs9_delta.json", "r") as file:
    full_outputs_delta = deserialize_full_outputs(json.load(file))

tilts_fine = np.arange(-np.pi / 2, np.pi / 2, step=np.pi / 500)
v3a = np.array([v3(1, 4, 1, 0, t) for t in tilts_fine])
w3a = np.array([w3(1, 4, 1, 0, t) for t in tilts_fine])
msea = (v3a - 1) ** 2 + w3a**2

v3b = np.array([v3(4, 1, 1, 0, t) for t in tilts_fine])
w3b = np.array([w3(4, 1, 1, 0, t) for t in tilts_fine])
mseb = (v3b - 1) ** 2 + w3b**2

v3c = np.array([v3(1, 1.5, 1, 0, t) for t in tilts_fine])
w3c = np.array([w3(1, 1.5, 1, 0, t) for t in tilts_fine])
msec = (v3c - 1) ** 2 + w3c**2

v3d = np.array([v3(1, 2, 1, 0, t) for t in tilts_fine])
w3d = np.array([w3(1, 2, 1, 0, t) for t in tilts_fine])
msed = (v3d - 1) ** 2 + w3d**2

ax[0].plot(tilts_fine, msea, color="blue", ls="--")
ax[0].plot(tilts_fine, mseb, color="red", ls="--")
ax[0].plot(tilts_fine, msec, color="green", ls="--")
ax[0].plot(tilts_fine, msed, color="black", ls="--")

average_confidence = get_confidence_for_slice(full_outputs, 1)
index_cc_fail = np.argmax(average_confidence < 0.25)
ax[0].fill_between(
    tilts9[average_confidence < 0.25], -0.5, 1.1, color="lightgray", alpha=0.5
)
ax[1].fill_between(
    tilts9[average_confidence < 0.25], -0.5, 1.1, color="lightgray", alpha=0.5
)

average_confidence = get_confidence_for_slice(full_outputs, 0)
index_cc_fail = np.argmax(np.logical_and(average_confidence < 0.25, tilts9 > 0))
ax[0].fill_between(
    tilts9[np.logical_and(average_confidence < 0.25, tilts9 < 0)],
    -0.5,
    1.1,
    color="lightgray",
    alpha=0.5,
    edgecolor="none",
)
ax[1].fill_between(
    tilts9[np.logical_and(average_confidence < 0.25, tilts9 < 0)],
    -0.5,
    1.1,
    color="lightgray",
    alpha=0.5,
    edgecolor="none",
)
ax[0].fill_between(
    [tilts9[index_cc_fail], np.pi / 2],
    -0.5,
    1.1,
    color="lightgray",
    alpha=0.5,
    edgecolor="none",
)
ax[1].fill_between(
    [tilts9[index_cc_fail], np.pi / 2],
    -0.5,
    1.1,
    color="lightgray",
    alpha=0.5,
    edgecolor="none",
)

labela = r"$\ell_\shortparallel/\ell_\perp=1/4$"
labelb = r"$\ell_\shortparallel/\ell_\perp=4$"
labelc = r"$\ell_\shortparallel/\ell_\perp=2/3$"
labeld = r"$\ell_\shortparallel/\ell_\perp=1/2$"

ax[0].scatter(
    tilts9,
    get_mse_for_slice(full_outputs, 0),
    color="blue",
)
ax[0].scatter(
    tilts9,
    get_mse_for_slice(full_outputs, 1),
    color="red",
)
ax[0].scatter(
    tilts9,
    get_mse_for_slice(full_outputs, 2),
    color="green",
)
ax[0].scatter(
    tilts9,
    get_mse_for_slice(full_outputs, 3),
    color="black",
)
ax[0].scatter(
    tilts9, get_mse_for_slice(full_outputs_delta, 0), color="blue", marker="v"
)
ax[0].scatter(tilts9, get_mse_for_slice(full_outputs_delta, 1), color="red", marker="v")
ax[0].hlines(0.1, np.min(tilts9), np.max(tilts9), color="black", ls="--", lw=0.5)

ax[1].scatter(tilts9, get_std_for_slice(full_outputs, 0), color="blue", label=labela)
ax[1].scatter(tilts9, get_std_for_slice(full_outputs, 1), color="red", label=labelb)
ax[1].scatter(tilts9, get_std_for_slice(full_outputs, 2), color="green", label=labelc)
ax[1].scatter(tilts9, get_std_for_slice(full_outputs, 3), color="black", label=labeld)
ax[1].scatter(
    tilts9, get_std_for_slice(full_outputs_delta, 0), color="blue", marker="v"
)
ax[1].scatter(tilts9, get_std_for_slice(full_outputs_delta, 1), color="red", marker="v")

ax[0].set_ylabel(r"$\text{MSE}$")
ax[1].set_ylabel(MSE)
ax[0].set_xlabel(r"$T/\tau_\text{d}$")
ax[1].set_xlabel(r"$T/\tau_\text{d}$")
ax[1].set_ylim(-0.1, 1)
ax[0].set_ylim(-0.1, 1)

ax[0].set_xlabel(r"$\theta$")
ax[0].set_xticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
ax[0].set_xticklabels([r"$-\pi/2$", r"$-\pi/4$", 0, r"$\pi/4$", r"$\pi/2$"])

ax[1].set_xlabel(r"$\theta$")
ax[1].set_xticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
ax[1].set_xticklabels([r"$-\pi/2$", r"$-\pi/4$", 0, r"$\pi/4$", r"$\pi/2$"])

ax[1].legend(loc=1)

plt.savefig("method_study_tilt.eps", bbox_inches="tight")

# Linear damping

fig, ax = cp.figure_multiple_rows_columns(1, 2)

with open("data/full_outputs10.json", "r") as file:
    full_outputs = deserialize_full_outputs(json.load(file))

print(
    "Plotting average over {} simulations".format(len(full_outputs[0][0].run_outputs))
)

plot_func(ax[0], taups10, full_outputs, func=get_mse_for_slice, mode=1)
plot_func(ax[1], taups10, full_outputs, func=get_std_for_slice, mode=1)

xlabel = r"$\tau_\shortparallel$"
ax[0].set_ylabel(r"$\text{MSE}$")
ax[1].set_ylabel(MSE)
ax[0].set_xlabel(xlabel)
ax[1].set_xlabel(xlabel)
ax[0].set_ylim(-0.1, 1)
ax[1].set_ylim(-0.1, 1)

plt.savefig("method_study_taup.eps", bbox_inches="tight")

# Correlated amplitudes and velocities

fig, ax = cp.figure_multiple_rows_columns(1, 2)

with open("data/full_outputs11.json", "r") as file:
    full_outputs = deserialize_full_outputs(json.load(file))

print(
    "Plotting average over {} simulations".format(len(full_outputs[0][0].run_outputs))
)

plot_func(ax[0], ss11, full_outputs, func=get_mse_for_slice, mode=1)
plot_func(ax[1], ss11, full_outputs, func=get_std_for_slice, mode=1)

inset_ax = inset_axes(
    ax[0],
    width="45%",
    height="45%",
    bbox_to_anchor=(0.125, -0.05, 1, 1),
    bbox_transform=ax[0].transAxes,
    loc="upper left",
)
vxs, vys = get_velocities(full_outputs, 0)
for i in np.arange(0, 10):
    inset_ax.scatter(ss11, np.array(vxs)[:, i], color="blue")

inset_ax.set_xticks([0, 1, 2])
inset_ax.set_xlabel(r"$\sigma$")
inset_ax.set_ylabel(r"$\widehat{v}$")
inset_ax.set_ylim(0, 2)

ax[0].set_ylabel(r"$\text{MSE}$")
ax[1].set_ylabel(MSE)
xlabel = r"$\sigma$"
ax[0].set_xlabel(xlabel)
ax[1].set_xlabel(xlabel)
ax[0].set_ylim(-0.1, 1)
ax[1].set_ylim(-0.1, 1)

plt.savefig("method_study_corr.eps", bbox_inches="tight")

# Random sizes

fig, ax = cp.figure_multiple_rows_columns(1, 2)

with open("data/full_outputs12.json", "r") as file:
    full_outputs = deserialize_full_outputs(json.load(file))

print(
    "Plotting average over {} simulations".format(len(full_outputs[0][0].run_outputs))
)

plot_func(ax[0], ss12, full_outputs, func=get_mse_for_slice, mode=1)
plot_func(ax[1], ss12, full_outputs, func=get_std_for_slice, mode=1)

ax[0].set_ylabel(r"$\text{MSE}$")
ax[1].set_ylabel(r"$\Delta \text{MSE}$")
xlabel = r"$\sigma$"
ax[0].set_xlabel(xlabel)
ax[1].set_xlabel(xlabel)
ax[0].set_ylim(-0.1, 1)
ax[1].set_ylim(-0.1, 1)

plt.savefig("method_study_rnd_sizes.eps", bbox_inches="tight")

# Analytical barberpole effect

lpara = 1
lperp = 4
v = 1
w = 0
a = np.arange(-np.pi / 2, np.pi / 2, step=np.pi / 500)

fig, axes = cp.figure_multiple_rows_columns(1, 2)

ax = axes[0]
values = 1 / taumax(1, 0, lpara, lperp, v, w, a)
ax.plot(a[values < 0], values[values < 0], label=r"$\wh{v}$", color="blue")
ax.plot(a[values > 0], values[values > 0], color="blue")

values = 1 / taumax(0, 1, lpara, lperp, v, w, a)
ax.plot(a[values < 0], values[values < 0], label=r"$\wh{w}$", color="red")
ax.plot(a[values > 0], values[values > 0], color="red")
ax.plot(a, v3(lpara, lperp, v, w, a), color="blue", ls="--")
ax.plot(a, w3(lpara, lperp, v, w, a), color="red", ls="--")

ax.set_ylabel(r"$\wh{v}, \, \wh{w}$")
ax.set_xlabel(r"$\alpha$")
ax.set_xticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
ax.set_xticklabels([r"$-\pi/2$", r"$-\pi/4$", 0, r"$\pi/4$", r"$\pi/2$"])
ax.legend(loc=4)
ax.set_ylim(-2, 2)
ax.text(s=r"$\ell_\shortparallel/\ell_\perp=1/4$", x=-1.6, y=1.6, size=6)

ax = axes[1]
lpara = 4
lperp = 1
values = 1 / taumax(1, 0, lpara, lperp, v, w, a)
ax.plot(a[values < 0], values[values < 0], label=r"$\wh{v}_2$", color="blue")
ax.plot(a[values > 0], values[values > 0], color="blue")

values = 1 / taumax(0, 1, lpara, lperp, v, w, a)
ax.plot(a[values < 0], values[values < 0], color="red")
ax.plot(a[values > 0], values[values > 0], color="red")
ax.plot(a, v3(lpara, lperp, v, w, a), label=r"$\wh{v}_3$", color="blue", ls="--")
ax.plot(a, w3(lpara, lperp, v, w, a), color="red", ls="--")

ax.set_ylabel(r"$\wh{v}, \, \wh{w}$")
ax.set_xlabel(r"$\alpha$")
ax.set_xticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
ax.set_xticklabels([r"$-\pi/2$", r"$-\pi/4$", 0, r"$\pi/4$", r"$\pi/2$"])
ax.legend(loc=3)
ax.set_ylim(-2, 2)
ax.text(s=r"$\ell_\shortparallel/\ell_\perp=4$", x=1, y=1.6, size=6)

plt.savefig("analytical_barberpole.eps", bbox_inches="tight")
