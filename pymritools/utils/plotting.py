import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc
import torch
import logging
import pathlib as plib
from pymritools.config.emc.params import SimulationData
from pymritools.utils.colormaps import check_colormap_available, get_colormap, show_available_colormaps

log_module = logging.getLogger(__name__)


def plot_gradient_pulse(
        pulse_x: torch.Tensor, pulse_y: torch.Tensor, grad_z: torch.Tensor,
        b1_vals: torch.Tensor, out_path: plib.Path | str, name: str):
    x_ax = torch.arange(pulse_x.shape[1])
    # calculate complex pulse shape
    p_cplx = pulse_x + 1j * pulse_y

    # set up figure
    fig = psub.make_subplots(
        rows=2, cols=1,
        specs=[[{"secondary_y": True}], [{}]]
    )
    if p_cplx.shape.__len__() > 1 and p_cplx.shape[-1] > 1:
        cmap = plc.sample_colorscale("Inferno", p_cplx.shape[0], 0.1, 0.9)
    else:
        cmap = plc.sample_colorscale("Inferno", 2, 0.6, 0.8)
    for i, p in enumerate(p_cplx):
        p_abs = torch.abs(p)
        fig.add_trace(
            go.Scatter(
                x=x_ax, y=p_abs,
                marker=dict(color=cmap[i]),
                showlegend=False
            ),
            row=1, col=1,
            secondary_y=False
        )

    fig.add_trace(
        go.Scatter(
            x=x_ax, y=p_cplx[0].angle(),
            showlegend=False, marker=dict(color="cadetblue"),
        ),
        row=1, col=1,
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], showlegend=False,
            marker=dict(
                size=15,
                cmin=b1_vals.min().item(),
                cmax=b1_vals.max().item(),
                colorscale="Inferno",
                colorbar=dict(title="B1+", thickness=10),
                showscale=True,
            )
        ),
        row=1, col=1,
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=x_ax, y=grad_z,
            marker=dict(color="cadetblue"),
            name="slice grad", fill="tozeroy", showlegend=False
        ),
        row=2, col=1
    )
    fig['layout']['title']['text'] = "Pulse - Gradient"
    fig.update_xaxes(title_text="sampling point")
    fig['layout']['yaxis']['title']['text'] = "magnitude [a.u.]"
    fig['layout']['yaxis2']['title']['text'] = "phase [pi]"
    fig['layout']['yaxis3']['title']['text'] = "gradient [mT/m]"

    out_path = plib.Path(out_path).absolute()
    fig_file = out_path.joinpath(f"plot_grad_pulse_{name}").with_suffix(".html")
    log_module.info(f"writing file: {fig_file.as_posix()}")
    fig.write_html(fig_file.as_posix())


def plot_signal_traces(sim_data: SimulationData, out_path: plib.Path | str, name: str = ""):
    if name:
        name = f"_{name}"
    # plot at most 5 values
    num_t2s = min(sim_data.t2_vals.shape[0], 5)
    num_b1s = min(sim_data.b1_vals.shape[0], 5)
    num_echoes = sim_data.signal_mag.shape[-1]
    t2_vals = sim_data.t2_vals[:num_t2s].numpy()
    b1_vals = sim_data.b1_vals[:num_b1s].numpy()
    # dims signal tensor [t1s, t2s, b1s, echoes, sim sampling pts]
    mag_plot_x = np.real(sim_data.signal_tensor[0, :, :, :num_echoes].numpy(force=True))
    mag_plot_y = np.imag(sim_data.signal_tensor[0, :, :, :num_echoes].numpy(force=True))
    mag_plot_abs = np.sqrt(mag_plot_y ** 2 + mag_plot_x ** 2)
    mag_plot_abs /= np.max(mag_plot_abs)
    mag_plot_phase = np.angle(sim_data.signal_tensor[0, :, :, :num_echoes].numpy(force=True)) / np.pi
    x_ax = np.arange(1, 1 + sim_data.signal_tensor.shape[-1])

    data = []
    echos = []
    label = []
    b1 = []
    t2 = []
    ax = []
    for idx_t2 in range(num_t2s):
        for idx_b1 in range(num_b1s):
            for idx_echo in range(num_echoes):
                trace_abs = mag_plot_abs[idx_t2, idx_b1, idx_echo]
                data.extend(trace_abs.tolist())
                label.extend(["mag"] * trace_abs.shape[0])
                trace_phase = mag_plot_phase[idx_t2, idx_b1, idx_echo]
                data.extend(trace_phase.tolist())
                label.extend(["phase"] * trace_abs.shape[0])

                echos.extend([idx_echo] * 2 * trace_abs.shape[0])
                b1.extend([b1_vals[idx_b1]] * 2 * trace_abs.shape[0])
                t2.extend([1000 * t2_vals[idx_t2]] * 2 * trace_abs.shape[0])
                ax.extend(x_ax.tolist() * 2)

    # fig.update_layout(legend_title_text="simulated signal curves")
    # fig.update_xaxes(title_text="virtual ADC sampling pt")
    # fig.update_yaxes(title_text="magnitude [normalized a.u.]", secondary_y=False)
    # fig.update_yaxes(title_text="phase [$pi]", secondary_y=True)

    df = pl.DataFrame({
        "data": data, "t2": t2, "b1": b1, "labels": label, "echos": echos, "ax": ax,
    })
    # fig = px.line(df, x="ax", y="data", color="labels", facet_row="t2", facet_col="b1", animation_frame="echos",
    #               labels=dict(x="sampling point", y="signal [a.u.]"))
    # out_path = plib.Path(out_path).absolute()
    # fig_file = out_path.joinpath(f"plot_signal_traces{name}").with_suffix(".html")
    # log_module.info(f"writing file: {fig_file.as_posix()}")
    # fig.write_html(fig_file.as_posix())


def plot_emc_sim_data(sim_data: SimulationData, out_path: plib.Path | str, name: str = ""):
    if name:
        name = f"_{name}"
    # plot at most 5 values
    num_t2s = min(sim_data.t2_vals.shape[0], 5)
    num_b1s = min(sim_data.b1_vals.shape[0], 5)
    t2_vals = sim_data.t2_vals[:num_t2s]
    b1_vals = sim_data.b1_vals[:num_b1s]
    # dims signal tensor [t1s, t2s, b1s, echoes, sim sampling pts]
    emc_mag = sim_data.signal_mag[0].clone().detach().cpu()
    emc_phase = sim_data.signal_phase[0].clone().detach().cpu()
    # setup figure
    fig = psub.make_subplots(
        rows=num_t2s, cols=num_b1s,
        subplot_titles=[f"T2: {t2_ * 1e3:.1f} ms, B1: {b1_:.1f}" for t2_ in t2_vals for b1_ in b1_vals],
        specs=[[{"secondary_y": True} for _ in b1_vals] for _ in t2_vals]
    )
    x_ax = torch.arange(1, 1 + sim_data.signal_mag.shape[-1])
    for idx_t2 in range(num_t2s):
        for idx_b1 in range(num_b1s):
            fig.add_trace(
                go.Scattergl(x=x_ax, y=emc_mag[idx_t2, idx_b1],
                           name=f"mag_emc"),
                row=1 + idx_t2, col=1 + idx_b1,
                secondary_y=False
            )
            fig.add_trace(
                go.Scattergl(x=x_ax, y=emc_phase[idx_t2, idx_b1] / torch.pi,
                           name=f"phase_emc"),
                row=1 + idx_t2, col=1 + idx_b1,
                secondary_y=True
            )
    fig.update_layout(legend_title_text="EMC simulated curves")
    fig.update_xaxes(title_text="# Echo")
    fig.update_yaxes(title_text="Signal magnitude", secondary_y=False)
    fig.update_yaxes(title_text="Signal phase [$pi]", secondary_y=True)

    out_path = plib.Path(out_path).absolute()
    fig_file = out_path.joinpath(f"plot_emc_signal{name}").with_suffix(".html")
    log_module.info(f"writing file: {fig_file.as_posix()}")
    fig.write_html(fig_file.as_posix())


# def plot_magnetization(mag_profile_df: pl.DataFrame, out_path: plib.Path | str,
#                        animate: bool = False, slice_thickness_mm: float = 0.0, name: str = ""):
#     if name:
#         name = f"_{name}"
#     if animate:
#         fig = px.line(
#             data_frame=mag_profile_df, x="axis", y="profile", color="dim",
#             animation_frame="name", labels={'y': 'Mag. Profile [a.u.]', 'x': 'sample axis [mm]'}
#         )
#
#     else:
#         fig = px.line(
#             data_frame=mag_profile_df, x="axis", y="profile", color="dim",
#             facet_col="name", facet_col_wrap=2, labels={'y': 'Mag. Profile [a.u.]', 'x': 'sample axis [mm]'}
#         )
#         fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
#         if slice_thickness_mm > 1e-3:
#             fig.add_vrect(x0=-slice_thickness_mm / 2, x1=slice_thickness_mm / 2,
#                           annotation_text="desired slice", annotation_position="bottom right",
#                           fillcolor="purple", opacity=0.25, line_width=0)
#     out_path = plib.Path(out_path).absolute()
#     fig_file = out_path.joinpath(f"plot_magnetization_propagation{name}").with_suffix(".html")
#     log_module.info(f"writing file: {fig_file.as_posix()}")
#     fig.write_html(fig_file.as_posix())


def plot_slice_img_tensor(slice_image_tensor: torch.Tensor, sim_data: SimulationData,
                          out_path: plib.Path | str, name: str = ""):
    if name:
        name = f"_{name}"
    # pick middle sim range values
    b1_idx = int(sim_data.b1_vals.shape[0] / 2)
    b1_val = f"{sim_data.b1_vals[b1_idx].numpy(force=True):.2f}".replace(".", "p")
    t2_idx = int(sim_data.t2_vals.shape[0] / 2)
    t2_val = f"{1000 * sim_data.t2_vals[t2_idx].numpy(force=True):.1f}ms".replace(".", "p")
    t1_idx = int(sim_data.t1_vals.shape[0] / 2)
    t1_val = f"{sim_data.t1_vals[t1_idx].numpy(force=True):.2f}s".replace(".", "p")
    name_extend = f"_t1-{t1_val}_t2-{t2_val}_b1-{b1_val}"
    # dims [num_echoes, slice_sampling_pts]
    slice_profile_sampling = slice_image_tensor[t1_idx, t2_idx, b1_idx].numpy(force=True)
    mag = np.abs(slice_profile_sampling.flatten())
    phase = np.angle(slice_profile_sampling.flatten())
    data = np.concatenate(
        (mag / np.max(mag), phase / np.pi),
        axis=0
    )
    echo_num = 1 * np.repeat(
            np.arange(slice_profile_sampling.shape[0]),
            slice_profile_sampling.shape[1],
            axis=0
        )
    echo_num = np.tile(echo_num, 2)
    axis = np.tile(
        1e3 * np.linspace(sim_data.sample_axis[0], sim_data.sample_axis[-1], slice_profile_sampling.shape[1]),
        slice_profile_sampling.shape[0]
    )
    axis = np.tile(axis, 2)
    labels = ["mag"] * np.prod(slice_profile_sampling.shape) + ["phase"] * np.prod(slice_profile_sampling.shape)
    # build df
    df = pl.DataFrame({
        "data": data,
        "echo_num": echo_num,
        "axis": axis,
        "label": labels
    })

    # plot
    # fig = px.line(data_frame=df, x="axis", y="data", color="echo_num",
    #               facet_row="label",
    #               labels={"x": "slice axis [mm]", "y": "profile [a.u.]", "color": "echo number"}
    #               )

    # out_path = plib.Path(out_path).absolute()
    # fig_file = out_path.joinpath(f"plot_slice_sampling_img_tensor{name}{name_extend}").with_suffix(".html")
    # log_module.info(f"writing file: {fig_file.as_posix()}")
    # fig.write_html(fig_file.as_posix())


def animation_volumetric_data(
        data_xyz: torch.Tensor, cmax: int,
        path_out: str | plib.Path, file_name: str,
        cmap: str = "Inferno", title: str = "",
        cmin: int = 0, width: int = 800, height: int = 800,
        x_ax_title: str = "A-P", y_ax_title: str = "R-L", z_ax_title: str = "I-S",
        file_suffix: str = "html", bg_color="#ffffff", template="plotly"):

    # check colormaps
    if check_colormap_available(name=cmap):
        colorscale = get_colormap(name=cmap)
    elif cmap in plc.named_colorscales():
        colorscale = plc.get_colorscale(cmap)
    else:
        msg = (f"colormap ({cmap}) not available as plotly colorscale or in the ScientificColorMap8 package. "
               f"Available scales:\nplotly :: {plc.named_colorscales()}\nscientific colorscale ::"
               f" {show_available_colormaps()}")
        log_module.error(msg)
        raise ValueError(msg)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            z=torch.zeros_like(data_xyz[..., 0]),
            colorscale=colorscale,
            cmin=cmin, cmax=cmax,
            colorbar=dict(thickness=20, ticklen=4, title=title)
        )
    )

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    nx, ny, nz = data_xyz.shape
    frames = [
        go.Frame(
            data=go.Surface(
                z=(nz - k) * torch.ones((nx, ny)),
                surfacecolor=data_xyz[:, :, -(k + 1)],
                cmin=cmin, cmax=cmax
            ),
            name=str(k)  # you need to name the frame for the animation to behave properly
        ) for k in range(nz)
    ]

    fig.update(frames=frames)

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]
    # Layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        scene=dict(
            zaxis=dict(
                range=[-0.1, 1.005 * nz], autorange=False,
                title=z_ax_title, backgroundcolor="#61615f", linecolor="#b0b0ae", gridcolor="#ccccca"
            ),
            xaxis=dict(title=x_ax_title, backgroundcolor="#61615f", linecolor="#b0b0ae", gridcolor="#ccccca"),
            yaxis=dict(title=y_ax_title, backgroundcolor="#61615f", linecolor="#b0b0ae", gridcolor="#ccccca"),
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        template=template,
        paper_bgcolor=bg_color,
        margin=dict(t=50, b=50, l=20, r=20),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(10)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )
    if "." not in file_suffix:
        file_suffix = f".{file_suffix}"

    fn = path_out.joinpath(file_name).with_suffix(file_suffix)
    log_module.info(f"Write file {fn}")

    if file_suffix == ".html":
        fig.write_html(fn)
    elif file_suffix in [".svg", ".png", ".pdf"]:
        fig.write_image(fn)
    else:
        msg = f"file suffix ({file_suffix}) not supported. Choose from (html, svg, png, pdf)"
        log_module.error(msg)
        raise ValueError(msg)
