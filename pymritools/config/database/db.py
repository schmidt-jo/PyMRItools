import logging
import pickle
import typing
from ..emc import EmcParameters, SimulationData
import numpy as np
import polars as pl
import pathlib as plib
from plotly.colors import sample_colorscale
import plotly.subplots as psub
import plotly.graph_objects as go
log_module = logging.getLogger(__name__)


class DB:
    def __init__(self, pl_dataframe: pl.DataFrame = pl.DataFrame(),
                 sequence_config: EmcParameters = EmcParameters(), name: str = "db_"):
        # define structure of pandas df
        self.indices: list = ["emc_mag", "emc_phase", "t2", "t1", "b1"]
        # check indices
        for ind in self.indices:
            if not ind in pl_dataframe.columns:
                err = f"db structure not given. Index {ind} not found. " \
                      f"Make sure these indices are columns in the dataframe: {self.get_indexes()};" \
                      f"\nIndices found in db: {pl_dataframe.columns}"
                log_module.error(err)
                raise ValueError(err)
        self.pl_dataframe: pl.DataFrame = pl_dataframe
        self.seq_params: EmcParameters = sequence_config
        self.np_mag_array: np.ndarray = np.array([*pl_dataframe["emc_mag"].to_numpy()])
        self.np_phase_array: np.ndarray = np.array([*pl_dataframe["emc_phase"].to_numpy()])
        self.etl: int = len(self.pl_dataframe["echo"].unique())
        # extract only name in case filename given
        name = plib.Path(name).absolute()
        name = name.stem
        self.name: str = name.__str__()

        # normalize
        # self.normalize()

    def get_indexes(self):
        return self.indices

    def get_t2_b1_values(self) -> (np.ndarray, np.ndarray):
        return np.unique(self.pl_dataframe["t2"]), np.unique(self.pl_dataframe["b1"])

    def plot(self,
             out_path: plib.Path | str, name: str = "",
             t1_range_s: tuple = None, t2_range_ms: tuple = (20, 50), b1_range: tuple = (0.5, 1.2),
             format: str = "html"):
        if name:
            name = f"_{name}"
        # select range
        df = self.pl_dataframe.clone()
        df["t2"] = 1e3 * df["t2"]
        df["t2"] = df["t2"].round(2)
        df["b1"] = df["b1"].round(2)
        df["echo"] = df["echo"] + 1
        if t2_range_ms is not None:
            df = df[(df["t2"] >= t2_range_ms[0]) & (df["t2"] < t2_range_ms[1])]
        if t1_range_s is not None:
            df = df[(df["t1"] >= t1_range_s[0]) & (df["t1"] < t1_range_s[1])]
        if b1_range is not None:
            df = df[(df["b1"] >= b1_range[0]) & (df["b1"] < b1_range[1])]
        # for now we only take one t1 value
        df = df[df["t1"] == df["t1"].unique()[0]].drop(columns=["t1"]).drop(columns="index").reset_index(drop=True)
        # setup colorscales to use
        c_scales = ["Purples", "Oranges", "Greens", "Blues"]
        echo_ax = df["echo"].to_numpy()
        # setup subplots
        while len(df["b1"].unique()) > len(c_scales):
            # drop randomly chosen b1 value
            b1_vals = df["b1"].unique().tolist()
            drop_idx = np.random.randint(len(b1_vals))
            b1_vals.pop(drop_idx)
            df = df[df["b1"].isin(b1_vals)]
        # setup subplots
        while len(df["t2"].unique()) > 12:
            # drop every second t2 value
            t2_vals = df["t2"].unique().tolist()[::2]
            df = df[df["t2"].isin(t2_vals)]
        num_plot_b1s = len(df["b1"].unique())
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
            c_tmp = sample_colorscale(c_scales[b1_idx], list(x))
            temp_df = df[df["b1"] == df["b1"].unique()[b1_idx]].reset_index(drop=True)
            for t2_idx in range(len(temp_df["t2"].unique())):
                t2 = temp_df["t2"].unique()[t2_idx]
                c = c_tmp[t2_idx]

                mag = temp_df[temp_df["t2"] == t2]["emc_mag"].to_numpy()
                mag /= np.abs(np.max(mag))
                fig.add_trace(
                    go.Scattergl(
                        x=echo_ax, y=mag, marker_color=c, showlegend=False
                    ),
                    1, 1
                )

                phase = temp_df[temp_df["t2"] == t2]["emc_phase"].to_numpy()
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
                        title=f"{df['b1'].unique()[b1_idx]}",
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
        if format == "html":
            fig_file = out_path.joinpath(f"emc_db{name}").with_suffix(".html")
            log_module.info(f"writing file: {fig_file.as_posix()}")
            fig.write_html(fig_file.as_posix())
        elif format in ["pdf", "svg", "png"]:
            fig_file = out_path.joinpath(f"emc_db{name}").with_suffix(f".{format}")
            log_module.info(f"writing file: {fig_file.as_posix()}")
            fig.write_image(fig_file.as_posix(), width=1200, height=800)
        else:
            err = f"Format {format} not recognized"
            log_module.error(err)
            raise AttributeError(err)

    def save(self, path: typing.Union[str, plib.Path]):
        path = plib.Path(path).absolute()
        if not path.suffixes:
            # given a path not a file
            path = path.joinpath(f"{self.name}_database_file.pkl")
        if ".pkl" not in path.suffixes:
            # given wrong filending
            log_module.info("filename saved as .pkl, adopting suffix.")
            path = path.with_suffix('.pkl')
        # mkdir ifn existent
        path.parent.mkdir(exist_ok=True, parents=True)

        log_module.info(f"writing file {path}")

        with open(path, "wb") as p_file:
            pickle.dump(self, p_file)

    @classmethod
    def load(cls, path: typing.Union[str, plib.Path]):
        path = plib.Path(path).absolute()
        if ".pkl" not in path.suffixes:
            # given wrong filending
            log_module.info("filename not .pkl, try adopting suffix.")
            path = path.with_suffix('.pkl')
        if not path.is_file():
            # given a path not a file
            err = f"{path.__str__()} not a file"
            log_module.error(err)
            raise ValueError(err)
        with open(path, "rb") as p_file:
            db = pickle.load(p_file)
        return db

    # def normalize(self):
    #     arr = self.np_mag_array
    #     norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    #     self.np_mag_array = np.divide(arr, norm, where=norm > 1e-12, out=np.zeros_like(arr))
    #
    #     for k in tqdm.trange(len(self.pd_dataframe), desc="normalizing db entries"):
    #         self.pd_dataframe.at[k, "emc_mag"] = self.np_mag_array[k]

    def get_numpy_array(self) -> (np.ndarray, np.ndarray):
        # self.normalize()
        return self.np_mag_array, self.np_phase_array

    def get_numpy_array_ids_t(self):
        np_mag, np_phase = self.get_numpy_array()
        np_mag = np.reshape(np_mag, (-1, self.etl))
        mag_norm = np.linalg.norm(np_mag, axis=-1, keepdims=True)
        np_mag = np.divide(np_mag, mag_norm, where=np_mag > 1e-12, out=np.zeros_like(np_mag))
        np_phase = np.reshape(np_phase, (-1, self.etl))
        return np_mag, np_phase, np.squeeze(mag_norm)

    def append_zeros(self):
        # want 0 lines for fitting noise
        # b1s = self.pd_dataframe.b1.unique().astype(float)
        # t1s = self.pd_dataframe.t1.unique().astype(float)
        # ds = self.pd_dataframe.d.unique().astype(float)
        # for b1, t1, d in [(b1_val, t1_val, d_val) for b1_val in b1s for t1_val in t1s for d_val in ds]:
        #     # when normalizing 0 curves will be left unchanged. Data curves are unlikely 0
        #     temp_row = self.pd_dataframe.iloc[0].copy()
        #     temp_row.emc_signal = np.full(len(temp_row.emc_signal), 1e-5)
        #     temp_row.t2 = 1e-3
        #     temp_row.b1 = b1
        #     temp_row.t1 = t1
        #     temp_row.d = d
        #     self.pd_dataframe.loc[len(self.pd_dataframe.index)] = temp_row
        #     # still append 0 curves that wont get scaled -> useful if normalization leaves signal curve flat
        #     temp_row = self.pd_dataframe.iloc[0].copy()
        #     temp_row.emc_signal = np.zeros([len(temp_row.emc_signal)])
        #     temp_row.t2 = 0.0
        #     temp_row.b1 = b1
        #     temp_row.t1 = t1
        #     temp_row.d = d
        #     self.pd_dataframe.loc[len(self.pd_dataframe.index)] = temp_row
        # self.pd_dataframe = self.pd_dataframe.reset_index(drop=True)
        # self.np_array = np.array([*self.pd_dataframe.emc_signal.to_numpy()])
        # self.normalize()
        # ToDo: needs to be reworked
        pass

    @classmethod
    def build_from_sim_data(cls, sim_params: EmcParameters, sim_data: SimulationData):
        d = {}
        index = 0
        for idx_t1 in range(sim_data.t1_vals.shape[0]):
            for idx_t2 in range(sim_data.t2_vals.shape[0]):
                for idx_b1 in range(sim_data.b1_vals.shape[0]):
                    for idx_echo in range(sim_data.signal_mag.shape[-1]):
                        td = {
                            "index": index,
                            "t1": sim_data.t1_vals[idx_t1].clone().detach().cpu().item(),
                            "t2": sim_data.t2_vals[idx_t2].clone().detach().cpu().item(),
                            "b1": sim_data.b1_vals[idx_b1].clone().detach().cpu().item(),
                            "echo": idx_echo,
                            "emc_mag": sim_data.signal_mag[
                                idx_t1, idx_t2, idx_b1, idx_echo].clone().detach().cpu().item(),
                            "emc_phase": sim_data.signal_phase[
                                idx_t1, idx_t2, idx_b1, idx_echo].clone().detach().cpu().item()
                        }
                        d.__setitem__(index, td)
                        index += 1
        db_pd = pl.DataFrame(d).T
        return cls(pl_dataframe=db_pd, sequence_config=sim_params)

    def get_total_num_curves(self) -> int:
        num_b1s = len(self.pl_dataframe["b1"].unique())
        num_t1s = len(self.pl_dataframe["t1"].unique())
        num_t2s = len(self.pl_dataframe["t2"].unique())
        return num_b1s * num_t2s * num_t1s


if __name__ == '__main__':
    dl = DB.load("test/test_db_database_file.pkl")
    dl.plot()
