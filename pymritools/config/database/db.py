import logging
import pickle

import torch

from pymritools.config.emc import EmcParameters, SimulationData
import numpy as np
import polars as pl
import pathlib as plib
from plotly.colors import sample_colorscale
import plotly.subplots as psub
import plotly.graph_objects as go
log_module = logging.getLogger(__name__)


class DB:
    """
    Database definition. We want this basically to be a polars DataFrame,
    to capture all signal magnitude and phase data and the respective simulation parameters, as well as echo number.
    For easier usage we want to define methods to easily extract appropriately sized
     numpy arrays and torch tensor for fitting and manipulation.
     We also attach the parameters used for creation to the object.
    """
    def __init__(self, data: pl.DataFrame = pl.DataFrame(), params: EmcParameters = EmcParameters()):
        self.indices: list = ["magnitude", "phase", "t2", "t1", "b1", "b0", "echo_num"]
        self.data: pl.DataFrame = data
        self.params: EmcParameters = params
        self._num_t1s: int = -1
        self._num_t2s: int = -1
        self._num_b1s: int = -1
        self._num_b0s: int = -1
        self._num_echoes: int = -1

    @classmethod
    def from_simulation_data(cls, params: EmcParameters, sim_data: SimulationData):
        """
        Takes simulation parameters and simulation data and builds database.
        :param params: EMC simulation parameters
        :param sim_data: simulated data object
        :return: DB
        """
        # we want to attach the respective values to the simulated data, dims: [t1s, t2s, b1s, b0s, etl]
        t2s = torch.tile(
            sim_data.t2_vals[None, :, None, None, None],
            (sim_data.num_t1s, 1, sim_data.num_b1s, sim_data.num_b0s, params.etl)
        )
        t1s = torch.tile(
            sim_data.t1_vals[:, None, None, None, None],
            (1, sim_data.num_t2s, sim_data.num_b1s, sim_data.num_b0s, params.etl)
        )
        b1s = torch.tile(
            sim_data.b1_vals[None, None, :, None, None],
            (sim_data.num_t1s, sim_data.num_t2s, 1, sim_data.num_b0s, params.etl)
        )
        b0s = torch.tile(
            sim_data.b0_vals[None, None, None, :, None],
            (sim_data.num_t1s, sim_data.num_t2s, sim_data.num_b1s, 1, params.etl)
        )
        echo_num = torch.tile(
            torch.arange(1, params.etl +1)[None, None, None, None],
            (sim_data.num_t1s, sim_data.num_t2s, sim_data.num_b1s, sim_data.num_b0s, 1)
        )
        df = pl.DataFrame({
            "index": torch.arange(sim_data.num_t1s * sim_data.num_t2s * sim_data.num_b1s * sim_data.num_b0s * params.etl).tolist(),
            "t1": t1s.flatten().tolist(),
            "t2": t2s.flatten().tolist(),
            "b1": b1s.flatten().tolist(),
            "b0": b0s.flatten().tolist(),
            "echo_num": echo_num.flatten().tolist(),
            "magnitude": sim_data.signal_mag.flatten().tolist(),
            "phase": sim_data.signal_phase.flatten().tolist()
        })
        instance = cls(data=df, params=params)
        instance._num_echoes = params.etl
        instance._num_t1s = sim_data.num_t1s
        instance._num_t2s = sim_data.num_t2s
        instance._num_b1s = sim_data.num_b1s
        instance._num_b0s = sim_data.num_b0s
        return instance

    def save(self, path: plib.Path | str):
        path = plib.Path(path).absolute()
        if not path.parent.is_dir():
            log_module.info(f"create path: {path.parent.as_posix()}")
            path.parent.mkdir(parents=True, exist_ok=True)
        if not ".pkl" in path.suffixes:
            log_module.info("filename not .pkl, try adopting suffix.")
            path = path.with_suffix('.pkl')
        log_module.info(f"writing file {path}")
        with open(path, "wb") as p_file:
            pickle.dump(self, p_file)

    @classmethod
    def load(cls, path: plib.Path | str):
        path = plib.Path(path).absolute()
        if not path.is_file():
            err = f"file {path.as_posix()} not found."
            log_module.error(err)
            raise FileNotFoundError(err)
        log_module.info(f"loading file {path}")
        with open(path, "rb") as p_file:
            instance = pickle.load(p_file)
        if not isinstance(instance, cls):
            err = f"decoded file not a class instance of {cls.__name__}"
            log_module.error(err)
            raise TypeError(err)
        return instance

    def get_indexes(self):
        return self.indices

    def get_t1_t2_b1_values(self, numpy: bool = False) -> (torch.tensor, torch.tensor, torch.tensor):
        """
        Returns torch tensors of used t1, t2 and b1 values of the simulation.
        :param numpy: toggle numpy output
        :return: (tensor t1, tensor t2, tensor b1)
        """
        t1 = self.data["t1"].unique().to_numpy()
        t2 = self.data["t2"].unique().to_numpy()
        b1 = self.data["b1"].unique().to_numpy()
        if numpy:
            return t1, t2, b1
        return torch.from_numpy(t1), torch.from_numpy(t2), torch.from_numpy(b1)

    def get_t1_t2_b1_b0_values(self, numpy: bool = False) -> (torch.tensor, torch.tensor, torch.tensor):
        """
        Returns torch tensors of used t1, t2 and b1 values of the simulation.
        :param numpy: toggle numpy output
        :return: (tensor t1, tensor t2, tensor b1)
        """
        t1 = self.data["t1"].unique().to_numpy()
        t2 = self.data["t2"].unique().to_numpy()
        b1 = self.data["b1"].unique().to_numpy()
        b0 = self.data["b0"].unique().to_numpy()
        if numpy:
            return t1, t2, b1, b0
        return torch.from_numpy(t1), torch.from_numpy(t2), torch.from_numpy(b1), torch.from_numpy(b0)

    def get_numpy_arrays_t1t2b1b0e(self):
        """
        Returns numpy arrays of magnitude and phase with dimensions [t1, t2, b1, echoes]
        :return: mag, phase
        """
        mag = self.data["magnitude"].to_numpy()
        mag = np.reshape(mag, (self._num_t1s, self._num_t2s, self._num_b1s, self._num_b0s, self._num_echoes))
        phase = self.data["phase"].to_numpy()
        phase = np.reshape(phase, (self._num_t1s, self._num_t2s, self._num_b1s, self._num_b0s, self._num_echoes))
        return mag, phase

    def get_torch_tensors_t1t2b1b0e(self):
        """
        Returns torch tensors of magnitude and phase with dimensions [t1, t2, b1, b0, echoes]
        :return: mag, phase
        """
        mag, phase = self.get_numpy_arrays_t1t2b1b0e()
        return torch.from_numpy(mag.copy()), torch.from_numpy(phase.copy())

    def plot(self,
             out_path: plib.Path | str, name: str = "",
             t1_range_s: tuple = None, t2_range_ms: tuple = (20, 50),
             b1_range: tuple = (0.5, 1.2),
             format: str = "html"):
        if name:
            name = f"_{name}"
        # select range
        if t1_range_s is None:
            t1_range_s = (self.data["t1"].min() - 0.1, self.data["t1"].max() + 0.1)
        if t2_range_ms is None:
            t2_range_ms = (self.data["t2"].min() - 0.1, self.data["t2"].max() + 0.1)
        if b1_range is None:
            b1_range = (self.data["b1"].min() - 0.1, self.data["b1"].max() + 0.1)
        # scale t2 to ms
        df = self.data.with_columns(pl.col("t2") * 1000)
        # take middle and last b0 value
        b0s = df["b0"].unique().to_numpy()
        b0_vals = [b0s[(b0s.shape[0] - 1) // 2]]
        if b0s.shape[0] > 1:
            b0_vals.append(b0s[-1])
        df = df.filter(
            (pl.col("t2").is_between(t2_range_ms[0], t2_range_ms[1])) &
            (pl.col("b1").is_between(b1_range[0], b1_range[1])) &
            (pl.col("t1").is_between(t1_range_s[0], t1_range_s[1])) &
            pl.col("b0").is_in(b0_vals)
        )

        # setup colorscales to use
        c_scales = ["Purples", "Oranges", "Greens", "Blues"]
        echo_ax = df["echo_num"].unique().to_numpy()

        # filter some data
        while len(df["b1"].unique()) > len(c_scales):
            # drop randomly chosen b1 value
            b1_vals = df["b1"].unique()
            b1_vals = b1_vals.sample(n=len(c_scales), with_replacement=False)
            df = df.filter(pl.col("b1").is_in(b1_vals.to_list()))
        # setup subplots
        while len(df["t2"].unique()) > 12:
            # drop every second t2 value
            t2_vals = df["t2"].unique().to_list()[::2]
            df = df.filter(pl.col("t2").is_in(t2_vals))

        num_plot_b1s = len(df["b1"].unique())

        for idx_b0, b0 in enumerate(b0_vals):
            df_b0 = df.filter(pl.col("b0").is_between(b0-1e-3, b0+1e-3))
            # setup figure
            titles = ["Magnitude", "Phase"]
            fig = psub.make_subplots(
                2, 1, shared_xaxes=True, subplot_titles=titles
            )

            x = np.linspace(0.2, 1, len(df["t2"].unique()))
            # edit axis labels
            fig['layout']['xaxis2']['title'] = 'Echo Number'
            fig['layout']['yaxis']['title'] = 'Signal [a.u.]'
            fig['layout']['yaxis2']['title'] = 'Phase [rad]'

            for b1_idx in range(num_plot_b1s):
                # set colorscale
                c_tmp = sample_colorscale(c_scales[b1_idx], list(x))
                # take subset of data
                temp_df = df_b0.filter(pl.col("b1") == df_b0["b1"].unique()[b1_idx])
                for t2_idx, t2 in enumerate(temp_df["t2"].unique().to_list()):
                    c = c_tmp[t2_idx]

                    mag = temp_df.filter(pl.col("t2") == t2)["magnitude"].to_numpy()
                    mag = mag / np.abs(np.max(mag))
                    fig.add_trace(
                        go.Scattergl(
                            x=echo_ax, y=mag, marker_color=c, showlegend=False
                        ),
                        1, 1
                    )

                    phase = temp_df.filter(pl.col("t2") == t2)["phase"].to_numpy()
                    fig.add_trace(
                        go.Scattergl(
                            x=echo_ax, y=phase, marker_color=c, showlegend=False
                        ),
                        2, 1
                    )
                if b1_idx == num_plot_b1s-1:
                    showticks = True
                else:
                    showticks = False
                # add colorbar
                colorbar_trace = go.Scattergl(
                    x=[None], y=[None], mode='markers',
                    showlegend=False,
                    marker=dict(
                        colorscale=c_scales[b1_idx], showscale=True,
                        cmin=t2_range_ms[0], cmax=t2_range_ms[1],
                        colorbar=dict(
                            title=f"{df['b1'].unique()[b1_idx]:.2f}",
                            x=1.02 + 0.05 * b1_idx,
                            showticklabels=showticks
                        ),
                    )
                )
                fig.add_trace(colorbar_trace, 1, 1)

            # colorbar labels
            fig.add_annotation(
                xref="x domain", yref="y domain", x=1.005, y=-0.5, showarrow=False,
                text="T2 [ms]", row=1, col=1, textangle=-90, font=dict(size=14)
            )
            fig.add_annotation(
                xref="x domain", yref="y domain", x=1.03, y=1.01, showarrow=False,
                text="B1+", row=1, col=1, textangle=0, font=dict(size=14)
            )

            out_path = plib.Path(out_path).absolute()
            fig_file = out_path.joinpath(f"emc_db{name}_b0-{b0}".replace(".", "p"))
            if format == "html":
                fig_file = fig_file.with_suffix(".html")
                log_module.info(f"writing file: {fig_file.as_posix()}")
                fig.write_html(fig_file.as_posix())
            elif format in ["pdf", "svg", "png"]:
                fig_file = fig_file.with_suffix(f".{format}")
                log_module.info(f"writing file: {fig_file.as_posix()}")
                fig.write_image(fig_file.as_posix(), width=1200, height=800)
            else:
                err = f"Format {format} not recognized"
                log_module.error(err)
                raise AttributeError(err)

    def get_total_num_curves(self) -> int:
        return self._num_t1s * self._num_t2s * self._num_b1s * self._num_b0s


if __name__ == '__main__':
    dl = DB.load("examples/simulation/results/database_test.pkl")
    dl.plot("examples/simulation/results/figs/emc_db.html")
