import numpy as np
import xarray as xr
import velocity_estimation.utils as u
import blobmodel as bm


class RunParameters:
    def __init__(
        self,
        vx,
        vy,
        T=1000,
        K=1000,
        delta=1,
        Lx=10,
        Ly=10,
        theta=0,
        taup=1e10,
        dt=0.01,
        bs_prop=bm.BlobShapeEnum.gaussian,
        bs_perp=bm.BlobShapeEnum.gaussian,
        wx=1,
        wy=1,
        blob_alignment=False,
        sigma=None,
        snr=None,
    ):
        self.vx = vx
        self.vy = vy
        self.T = T
        self.K = K
        self.delta = delta
        self.Lx = Lx
        self.Ly = Ly
        self.theta = theta
        self.taup = taup
        self.dt = dt
        self.bs_prop = bs_prop
        self.bs_perp = bs_perp
        self.wx = wx
        self.wy = wy
        self.blob_alignment = blob_alignment
        self.sigma = sigma
        self.snr = snr

    def to_dict(self):
        data = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.generic):
                data[k] = v.item()  # convert NumPy scalars to Python scalars
            elif isinstance(v, bm.BlobShapeEnum):
                data[k] = v.name  # store enum by name
            else:
                data[k] = v
        return data

    @classmethod
    def from_dict(cls, dict_params):
        # Convert enum fields from their string names back to enum members
        if "bs_prop" in dict_params and isinstance(dict_params["bs_prop"], str):
            dict_params["bs_prop"] = bm.BlobShapeEnum[dict_params["bs_prop"]]
        if "bs_perp" in dict_params and isinstance(dict_params["bs_perp"], str):
            dict_params["bs_perp"] = bm.BlobShapeEnum[dict_params["bs_perp"]]

        return cls(**dict_params)


class RunOutput:
    def __init__(self, out_vx, out_vy, confidence):
        self.out_vx = out_vx
        self.out_vy = out_vy
        self.confidence = confidence

    def to_dict(self):
        return {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, dict_output):
        return cls(**dict_output)


class RunResults:
    def __init__(self, run_params: RunParameters, run_outputs):
        self.run_params = run_params
        self.run_outputs = run_outputs

    def to_dict(self):
        return {
            "run_params": self.run_params.to_dict(),
            "run_outputs": [output.to_dict() for output in self.run_outputs],
        }

    @classmethod
    def from_dict(cls, dict_results):
        run_params = RunParameters.from_dict(dict_results["run_params"])
        run_outputs = [
            RunOutput.from_dict(output) for output in dict_results["run_outputs"]
        ]
        return cls(run_params, run_outputs)


class NoisyImagingDataInterface(u.SyntheticBlobImagingDataInterface):
    def __init__(self, ds: xr.Dataset, snr: float):
        self.ds = ds
        self.snr = snr

    def get_signal(self, x: int, y: int) -> np.ndarray:
        signal = self.ds.isel(x=x, y=y)["n"].values
        std = signal.std()
        return signal + 1 / self.snr * np.random.normal(0, std, size=len(signal))
