from blobmodel import BlobFactory, Blob, AbstractBlobShape, DistributionEnum
import numpy as np


class RandomVelocityBlobFactory(BlobFactory):
    def __init__(
        self,
        A_dist: str = DistributionEnum.exp,
        wx_dist: str = DistributionEnum.deg,
        wy_dist: str = DistributionEnum.deg,
        vx_dist: str = DistributionEnum.deg,
        vy_dist: str = DistributionEnum.deg,
        spx_dist: str = DistributionEnum.deg,
        spy_dist: str = DistributionEnum.deg,
        A_parameter: float = 1.0,
        wx_parameter: float = 1.0,
        wy_parameter: float = 1.0,
        vx_parameter: float = 1.0,
        vy_parameter: float = 1.0,
        shape_param_x_parameter: float = 0.5,
        shape_param_y_parameter: float = 0.5,
        blob_alignment: bool = True,
        sigma: float = 0,
    ) -> None:
        self.amplitude_dist = A_dist
        self.width_x_dist = wx_dist
        self.width_y_dist = wy_dist
        self.velocity_x_dist = vx_dist
        self.velocity_y_dist = vy_dist
        self.shape_param_x_dist = spx_dist
        self.shape_param_y_dist = spy_dist
        self.amplitude_parameter = A_parameter
        self.width_x_parameter = wx_parameter
        self.width_y_parameter = wy_parameter
        self.velocity_x_parameter = vx_parameter
        self.velocity_y_parameter = vy_parameter
        self.shape_param_x_parameter = shape_param_x_parameter
        self.shape_param_y_parameter = shape_param_y_parameter
        self.blob_alignment = blob_alignment
        self.theta_setter = lambda: None
        self.sigma = sigma

    def _draw_random_variables(
        self,
        dist_type: str,
        free_parameter: float,
        num_blobs: int,
    ) -> np.ndarray:
        if dist_type == DistributionEnum.exp:
            return np.random.exponential(scale=1, size=num_blobs).astype(np.float64)
        elif dist_type == DistributionEnum.deg:
            return free_parameter * np.ones(num_blobs).astype(np.float64)
        elif dist_type == DistributionEnum.zeros:
            return np.zeros(num_blobs).astype(np.float64)
        else:
            raise NotImplementedError(
                self.__class__.__name__ + ".distribution function not implemented"
            )

    def sample_blobs(
        self,
        Ly: float,
        T: float,
        num_blobs: int,
        blob_shape: AbstractBlobShape,
        t_drain,
    ):
        amps = self._draw_random_variables(
            dist_type=self.amplitude_dist,
            free_parameter=self.amplitude_parameter,
            num_blobs=num_blobs,
        )
        wxs = self._draw_random_variables(
            self.width_x_dist, self.width_x_parameter, num_blobs
        )
        wys = self._draw_random_variables(
            self.width_y_dist, self.width_y_parameter, num_blobs
        )
        vxs = self.velocity_x_parameter + np.random.uniform(
            low=-self.sigma / 2, high=self.sigma / 2, size=num_blobs
        ).astype(np.float64)
        vys = self.velocity_y_parameter + np.random.uniform(
            low=-self.sigma / 2, high=self.sigma / 2, size=num_blobs
        ).astype(np.float64)
        spxs = self._draw_random_variables(
            self.shape_param_x_dist, self.shape_param_x_parameter, num_blobs
        )
        spys = self._draw_random_variables(
            self.shape_param_y_dist, self.shape_param_y_parameter, num_blobs
        )
        # For now, only a lambda parameter is implemented
        spxs_dict = [{"lam": s} for s in spxs]
        spys_dict = [{"lam": s} for s in spys]
        posxs = np.zeros(num_blobs)
        posys = np.random.uniform(low=0.0, high=Ly, size=num_blobs)
        t_inits = np.random.uniform(low=0, high=T, size=num_blobs)

        blobs = [
            Blob(
                blob_id=i,
                blob_shape=blob_shape,
                amplitude=amps[i],
                width_prop=wxs[i],
                width_perp=wys[i],
                v_x=vxs[i],
                v_y=vys[i],
                pos_x=posxs[i],
                pos_y=posys[i],
                t_init=t_inits[i],
                t_drain=t_drain,
                prop_shape_parameters=spxs_dict[i],
                perp_shape_parameters=spys_dict[i],
                blob_alignment=self.blob_alignment,
                theta=self.theta_setter(),
            )
            for i in range(num_blobs)
        ]

        # sort blobs by amplitude
        return sorted(blobs, key=lambda x: x.amplitude)

    def set_theta_setter(self, theta_setter):
        self.theta_setter = theta_setter

    def is_one_dimensional(self) -> bool:
        return self.velocity_y_dist == "zeros"


class CorrelatedBlobFactory(BlobFactory):
    def __init__(
        self,
        A_dist: DistributionEnum = DistributionEnum.exp,
        wx_dist: DistributionEnum = DistributionEnum.deg,
        wy_dist: DistributionEnum = DistributionEnum.deg,
        vx_dist: DistributionEnum = DistributionEnum.deg,
        vy_dist: DistributionEnum = DistributionEnum.normal,
        spx_dist: DistributionEnum = DistributionEnum.deg,
        spy_dist: DistributionEnum = DistributionEnum.deg,
        A_parameter: float = 1.0,
        wx_parameter: float = 1.0,
        wy_parameter: float = 1.0,
        vx_parameter: float = 1.0,
        vy_parameter: float = 1.0,
        shape_param_x_parameter: float = 0.5,
        shape_param_y_parameter: float = 0.5,
        blob_alignment: bool = True,
        s: float = 0,
    ) -> None:
        self.amplitude_dist = A_dist
        self.width_x_dist = wx_dist
        self.width_y_dist = wy_dist
        self.velocity_x_dist = vx_dist
        self.velocity_y_dist = vy_dist
        self.shape_param_x_dist = spx_dist
        self.shape_param_y_dist = spy_dist
        self.amplitude_parameter = A_parameter
        self.width_x_parameter = wx_parameter
        self.width_y_parameter = wy_parameter
        self.velocity_x_parameter = vx_parameter
        self.velocity_y_parameter = vy_parameter
        self.shape_param_x_parameter = shape_param_x_parameter
        self.shape_param_y_parameter = shape_param_y_parameter
        self.blob_alignment = blob_alignment
        self.theta_setter = lambda: None
        self.s = s

    def _draw_random_variables(
        self,
        dist_type: DistributionEnum,
        free_parameter: float,
        num_blobs: int,
    ) -> np.ndarray:
        if dist_type == DistributionEnum.exp:
            return np.random.exponential(scale=1, size=num_blobs).astype(np.float64)
        elif dist_type == DistributionEnum.normal:
            return np.random.normal(loc=0, scale=free_parameter, size=num_blobs).astype(
                np.float64
            )
        elif dist_type == DistributionEnum.uniform:
            return np.random.uniform(
                low=1 - free_parameter / 2, high=1 + free_parameter / 2, size=num_blobs
            ).astype(np.float64)
        elif dist_type == DistributionEnum.rayleigh:
            return np.random.rayleigh(
                scale=np.sqrt(2.0 / np.pi), size=num_blobs
            ).astype(np.float64)
        elif dist_type == DistributionEnum.deg:
            return free_parameter * np.ones(num_blobs).astype(np.float64)
        elif dist_type == DistributionEnum.zeros:
            return np.zeros(num_blobs).astype(np.float64)
        else:
            raise NotImplementedError(
                self.__class__.__name__ + ".distribution function not implemented"
            )

    def sample_blobs(
        self,
        Ly: float,
        T: float,
        num_blobs: int,
        blob_shape: AbstractBlobShape,
        t_drain,
    ):
        amps = self._draw_random_variables(
            dist_type=DistributionEnum.uniform,
            free_parameter=self.s,
            num_blobs=num_blobs,
        )
        wxs = self._draw_random_variables(
            self.width_x_dist, self.width_x_parameter, num_blobs
        )
        wys = self._draw_random_variables(
            self.width_y_dist, self.width_y_parameter, num_blobs
        )
        vxs = amps * self.velocity_x_parameter
        vys = amps * self.velocity_y_parameter
        spxs = self._draw_random_variables(
            self.shape_param_x_dist, self.shape_param_x_parameter, num_blobs
        )
        spys = self._draw_random_variables(
            self.shape_param_y_dist, self.shape_param_y_parameter, num_blobs
        )
        # For now, only a lambda parameter is implemented
        spxs_dict = [{"lam": s} for s in spxs]
        spys_dict = [{"lam": s} for s in spys]
        posxs = np.zeros(num_blobs)
        posys = np.random.uniform(low=0.0, high=Ly, size=num_blobs)
        t_inits = np.random.uniform(low=0, high=T, size=num_blobs)

        blobs = [
            Blob(
                blob_id=i,
                blob_shape=blob_shape,
                amplitude=amps[i],
                width_prop=wxs[i],
                width_perp=wys[i],
                v_x=vxs[i],
                v_y=vys[i],
                pos_x=posxs[i],
                pos_y=posys[i],
                t_init=t_inits[i],
                t_drain=t_drain,
                prop_shape_parameters=spxs_dict[i],
                perp_shape_parameters=spys_dict[i],
                blob_alignment=self.blob_alignment,
                theta=self.theta_setter(),
            )
            for i in range(num_blobs)
        ]

        # sort blobs by amplitude
        return sorted(blobs, key=lambda x: x.amplitude)

    def set_theta_setter(self, theta_setter):
        self.theta_setter = theta_setter

    def is_one_dimensional(self) -> bool:
        return self.velocity_y_dist == "zeros"


class RandomBlobFactory(BlobFactory):
    def __init__(
        self,
        A_dist: str = "exp",
        wx_dist: str = "deg",
        wy_dist: str = "deg",
        vx_dist: str = "deg",
        vy_dist: str = "normal",
        spx_dist: str = "deg",
        spy_dist: str = "deg",
        A_parameter: float = 1.0,
        wx_parameter: float = 1.0,
        wy_parameter: float = 1.0,
        vx_parameter: float = 1.0,
        vy_parameter: float = 1.0,
        shape_param_x_parameter: float = 0.5,
        shape_param_y_parameter: float = 0.5,
        blob_alignment: bool = True,
        s: float = 0,
    ) -> None:
        self.amplitude_dist = A_dist
        self.width_x_dist = wx_dist
        self.width_y_dist = wy_dist
        self.velocity_x_dist = vx_dist
        self.velocity_y_dist = vy_dist
        self.shape_param_x_dist = spx_dist
        self.shape_param_y_dist = spy_dist
        self.amplitude_parameter = A_parameter
        self.width_x_parameter = wx_parameter
        self.width_y_parameter = wy_parameter
        self.velocity_x_parameter = vx_parameter
        self.velocity_y_parameter = vy_parameter
        self.shape_param_x_parameter = shape_param_x_parameter
        self.shape_param_y_parameter = shape_param_y_parameter
        self.blob_alignment = blob_alignment
        self.theta_setter = lambda: None
        self.s = s

    def _draw_random_variables(
        self,
        dist_type: str,
        free_parameter: float,
        num_blobs: int,
    ) -> np.ndarray:
        if dist_type == "exp":
            return np.random.exponential(scale=1, size=num_blobs).astype(np.float64)
        elif dist_type == "gamma":
            return np.random.gamma(
                shape=free_parameter, scale=1 / free_parameter, size=num_blobs
            ).astype(np.float64)
        elif dist_type == "normal":
            return np.random.normal(loc=0, scale=free_parameter, size=num_blobs).astype(
                np.float64
            )
        elif dist_type == "uniform":
            return np.random.uniform(
                low=1 - free_parameter / 2, high=1 + free_parameter / 2, size=num_blobs
            ).astype(np.float64)
        elif dist_type == "ray":
            return np.random.rayleigh(
                scale=np.sqrt(2.0 / np.pi), size=num_blobs
            ).astype(np.float64)
        elif dist_type == "deg":
            return free_parameter * np.ones(num_blobs).astype(np.float64)
        elif dist_type == "zeros":
            return np.zeros(num_blobs).astype(np.float64)
        else:
            raise NotImplementedError(
                self.__class__.__name__ + ".distribution function not implemented"
            )

    def sample_blobs(
        self,
        Ly: float,
        T: float,
        num_blobs: int,
        blob_shape: AbstractBlobShape,
        t_drain,
    ):
        amps = self._draw_random_variables(
            self.amplitude_dist, self.amplitude_parameter, num_blobs
        )
        wxs = np.random.uniform(low=1 - self.s / 2, high=1 + self.s / 2, size=num_blobs)
        wys = np.random.uniform(low=1 - self.s / 2, high=1 + self.s / 2, size=num_blobs)

        vxs = self._draw_random_variables(
            self.velocity_x_dist, self.velocity_x_parameter, num_blobs
        )
        vys = self._draw_random_variables(
            self.velocity_y_dist, self.velocity_y_parameter, num_blobs
        )
        spxs = self._draw_random_variables(
            self.shape_param_x_dist, self.shape_param_x_parameter, num_blobs
        )
        spys = self._draw_random_variables(
            self.shape_param_y_dist, self.shape_param_y_parameter, num_blobs
        )
        # For now, only a lambda parameter is implemented
        spxs_dict = [{"lam": s} for s in spxs]
        spys_dict = [{"lam": s} for s in spys]
        posxs = np.zeros(num_blobs)
        posys = np.random.uniform(low=0.0, high=Ly, size=num_blobs)
        t_inits = np.random.uniform(low=0, high=T, size=num_blobs)

        blobs = [
            Blob(
                blob_id=i,
                blob_shape=blob_shape,
                amplitude=amps[i],
                width_prop=wxs[i],
                width_perp=wys[i],
                v_x=vxs[i],
                v_y=vys[i],
                pos_x=posxs[i],
                pos_y=posys[i],
                t_init=t_inits[i],
                t_drain=t_drain,
                prop_shape_parameters=spxs_dict[i],
                perp_shape_parameters=spys_dict[i],
                blob_alignment=self.blob_alignment,
                theta=self.theta_setter(),
            )
            for i in range(num_blobs)
        ]

        # sort blobs by amplitude
        return sorted(blobs, key=lambda x: x.amplitude)

    def set_theta_setter(self, theta_setter):
        self.theta_setter = theta_setter

    def is_one_dimensional(self) -> bool:
        return self.velocity_y_dist == DistributionEnum.zeros
