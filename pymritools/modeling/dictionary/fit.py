import pathlib as plib
import nibabel as nib
import torch
import numpy as np
from torch import nn
import tqdm
import json
from emc_torch import DB
from emc_torch.fitting import options, io
import plotly.graph_objects as go
import plotly.subplots as psub
import logging
import scipy.interpolate as scinterp

log_module = logging.getLogger(__name__)
logging.getLogger('simple_parsing').setLevel(logging.WARNING)



class BruteForce:
    def __init__(self, slice_signal: torch.tensor,
                 db_torch_mag: torch.tensor, db_t2s_s: torch.tensor, db_b1s: torch.tensor,
                 delta_t_r2p_ms: torch.tensor, device: torch.device = torch.device("cpu"),
                 r2p_range_Hz: tuple = (0.001, 200), r2p_sampling_size: int = 220, slice_b1: torch.tensor = None,
                 b1_weighting_factor: float = 0.5):
        log_module.info("Brute Force matching algorithm")
        # save some vars
        # torch gpu processing
        self.device = device
        log_module.info(f"Set device: {device}")
        self.batch_size: int = 2000

        self.nx, self.ny, self.etl = slice_signal.shape
        self.delta_t_r2p_s: torch.tensor = 1e-3 * delta_t_r2p_ms.to(self.device)
        # database and t2 and b1 values
        self.db_t2s_s_unique: torch.tensor = torch.unique(db_t2s_s).to(self.device)
        self.num_t2s: int = self.db_t2s_s_unique.shape[0]
        self.db_b1s_unique: torch.tensor = torch.unique(db_b1s).to(self.device)
        self.db_delta_b1: torch.tensor = torch.diff(self.db_b1s_unique)[0]
        self.num_b1s: int = self.db_b1s_unique.shape[0]
        self.db_mag: torch.tensor = db_torch_mag.to(self.device)
        self.db_mag_shaped = torch.reshape(db_torch_mag, (self.num_t2s, self.num_b1s, self.etl)).to(self.device)

        # take signal as input
        self.signal = torch.reshape(slice_signal, (-1, self.etl)).to(self.device)

        self.b1: torch.tensor = slice_b1
        if self.b1 is not None:
            self.b1w = True
            self.b1 = torch.flatten(self.b1).to(self.device)
            self.b1w_factor = b1_weighting_factor
        else:
            self.b1w = False
            self.b1w_factor = 0.0

        # set up SE based signal
        self.se_select = torch.abs(self.delta_t_r2p_s) < 1e-9
        # norm
        self.se_signal_norm = torch.linalg.norm(self.signal[:, self.se_select], dim=-1, keepdim=True)
        self.se_db_mag_norm = torch.linalg.norm(self.db_mag_shaped[:, :, self.se_select], dim=-1, keepdim=True)

        # normalize both full echo trains based on the SE norms, so the se data coincides
        self.db_mag_shaped = torch.nan_to_num(
            torch.divide(self.db_mag_shaped, self.se_db_mag_norm),
            nan=0.0, posinf=0.0
        )
        self.signal = torch.nan_to_num(
            torch.divide(self.signal, self.se_signal_norm),
            nan=0.0, posinf=0.0
        )

        self.estimates_t2 = torch.zeros((self.nx * self.ny), dtype=self.db_t2s_s_unique.dtype)
        self.l2_error = torch.zeros((self.nx * self.ny), dtype=self.db_t2s_s_unique.dtype)
        self.estimates_r2p = torch.zeros((self.nx * self.ny), dtype=self.db_t2s_s_unique.dtype)
        self.estimates_b1 = torch.zeros((self.nx * self.ny), dtype=self.db_b1s_unique.dtype)
        self.b1_rmse = torch.zeros((self.nx * self.ny), dtype=self.db_b1s_unique.dtype)
        # need to reduce batch size due to memory constraints
        self.batch_size: int = 50

    def estimate_values(self):
        # need to batch data of whole slice
        num_batches = int(np.ceil(self.signal.shape[0] / self.batch_size))
        for idx_batch in tqdm.trange(num_batches, desc="match dictionary"):
            start_idx = idx_batch * self.batch_size
            end_idx = np.min([(idx_batch + 1) * self.batch_size, self.signal.shape[0]])
            # load curves
            data_batch = self.signal[start_idx:end_idx]
            # match b1 if provided
            if self.b1w:
                # b1 weighting, we have a dictionary per data point reduced to the closest b1 value
                b1_cost = torch.square(self.b1[None, start_idx:end_idx] - self.db_b1s_unique[:, None])
                b1_min = torch.min(b1_cost, dim=0)
                self.b1_rmse[start_idx:end_idx] = torch.sqrt(b1_min.values)
                b1_min = b1_min.indices

                b1_db = self.db_mag_shaped[:, b1_min]
                # match only SE data
                # data dims [bs, t], db dims [t2, bs, t]
                # l2
                l2_t2_b1 = torch.linalg.norm(data_batch[None, :, self.se_select] - b1_db[:, :, self.se_select], dim=-1)
                # get indices of minima along each dim
                fit_min = torch.min(l2_t2_b1, dim=0)
                self.l2_error[start_idx:end_idx] = fit_min.values
                # dot
                # dot_t2_b1 = torch.sum(data_batch[None, :, self.se_select] * b1_db[:, :, self.se_select], dim=-1)
                # # get indices of maxima along each dim
                # fit_max = torch.max(dot_t2_b1, dim=0)
                # self.l2_error[start_idx:end_idx] = fit_max.values
                # take b1 indices from above
                fit_idx = fit_min.indices
                fit_idx = torch.concatenate((fit_idx[:, None], b1_min[:, None]), dim=1)
            else:
                # data dims [bs, t], db dims [t2, b1, t]
                l2_t2_b1 = torch.linalg.norm(
                    data_batch[None, None, self.se_select] - self.db_mag_shaped[:, :, None, self.se_select],
                    dim=-1
                )
                # self.l2_error = torch.min(l2_t2_b1, dim=(0, 1)).values
                fit_idx = torch.squeeze(
                    torch.stack(
                        [(l2_t2_b1[:, i] == torch.min(l2_t2_b1[:, :, i])).nonzero() for i in range(l2_t2_b1.shape[-1])]
                    )
                ).detach().cpu()
            self.estimates_b1[start_idx:end_idx] = self.db_b1s_unique[fit_idx[:, 1]]
            self.estimates_t2[start_idx:end_idx] = self.db_t2s_s_unique[fit_idx[:, 0]]

            # find r2p for non SE data
            # set extracted db
            db_t2_b1 = self.db_mag_shaped[fit_idx[:, 0], fit_idx[:, 1]][:, ~self.se_select]
            # signal and db normalized to match se data points
            signal = data_batch[:, ~self.se_select]
            # minimize error
            log_db_sig = torch.log(db_t2_b1) - torch.log(signal)
            r2p = log_db_sig / self.delta_t_r2p_s[None, ~self.se_select]
            self.estimates_r2p[start_idx:end_idx] = torch.mean(r2p, dim=-1)

    def get_maps(self):
        t2_ms = 1e3 * torch.reshape(self.estimates_t2, (self.nx, self.ny))
        r2p = torch.reshape(self.estimates_r2p, (self.nx, self.ny))
        b1 = torch.reshape(self.estimates_b1, (self.nx, self.ny))
        err_l2 = torch.reshape(self.l2_error, (self.nx, self.ny))
        rmse_b1 = torch.reshape(self.b1_rmse, (self.nx, self.ny))
        return t2_ms.detach().cpu(), r2p.detach().cpu(), b1.detach().cpu(), err_l2.detach().cpu(), rmse_b1.detach().cpu()


def plot_loss(losses: list, save_path: plib.Path, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(x=np.arange(losses.__len__()), y=losses)
    )
    fig.update_layout(width=800, height=500)
    fig_name = save_path.joinpath(title).with_suffix(".html")
    log_module.info(f"write file: {fig_name}")
    fig.write_html(fig_name.as_posix())


def plot_maps(t2: torch.tensor, r2p: torch.tensor, b1: torch.tensor,
              t2_init: torch.tensor, b1_init: torch.tensor,
              save_path: plib.Path, title: str):
    fig = psub.make_subplots(
        rows=1, cols=5, shared_xaxes=True, shared_yaxes=True,
        column_titles=["T2 [ms]", "R2p [Hz]", "B1+", "T2 init [ms]", "B1+ init"],
        horizontal_spacing=0.01,
        vertical_spacing=0
    )
    zmin = [0, 0, 0.2, 0, 0.2]
    data_list = [t2, r2p, b1, t2_init, b1_init]
    for idx_data in range(len(data_list)):
        data = data_list[idx_data].numpy(force=True)

        fig.add_trace(
            go.Heatmap(
                z=data, transpose=True, zmin=zmin[idx_data], zmax=np.max(data),
                showscale=False, colorscale="Magma"
            ),
            row=1, col=1 + idx_data
        )
        if idx_data > 0:
            x = f"x{idx_data + 1}"
        else:
            x = "x"
        fig.update_xaxes(visible=False, row=1, col=1 + idx_data)
        fig.update_yaxes(visible=False, row=1, col=1 + idx_data, scaleanchor=x)

    fig.update_layout(
        width=1000, height=500
    )
    fig_name = save_path.joinpath(title).with_suffix(".html")
    log_module.info(f"write file: {fig_name}")
    fig.write_html(fig_name.as_posix())


def megesse_fit(
        fit_config: options.FitConfig, data_nii: torch.tensor, db_torch_mag: torch.tensor,
        db: DB, name: str, b1_nii=None):
    # data_nii = data_nii[:, :, 17:19, :]
    if b1_nii is not None:
        b1_scale = fit_config.b1_tx_scale
        if torch.max(b1_nii) > 10:
            b1_nii = b1_nii / 100
        b1_nii *= b1_scale
    ep_path = plib.Path(fit_config.echo_props_path).absolute()
    if not ep_path.is_file():
        err = f"echo properties file: {ep_path} not found or not a file."
        log_module.error(err)
        raise FileNotFoundError(err)
    log_module.info(f"loading echo property file: {ep_path}")
    with open(ep_path.as_posix(), "r") as j_file:
        echo_props = json.load(j_file)

    if echo_props.__len__() < db_torch_mag.shape[-1]:
        warn = "echo type list not filled or shorter than database etl. filling with SE type acquisitions"
        log_module.warning(warn)
    while echo_props.__len__() < db_torch_mag.shape[-1]:
        # if the list is too short or insufficiently filled we assume SE acquisitions
        echo_props[echo_props.__len__()] = {"type": "SE", "te_ms": 0.0, "time_to_adj_se_ms": 0.0}
    # possibly need some kind of convenient class to coherently store information
    # need tensor to hold time deltas to SE
    delta_t_ms_to_se = []
    for idx_e in range(echo_props.__len__()):
        delta_t_ms_to_se.append((echo_props[str(idx_e)]["time_to_adj_se_ms"]))
    delta_t_ms_to_se = torch.abs(torch.tensor(delta_t_ms_to_se))

    # get values
    t2_vals = torch.from_numpy(db.pd_dataframe[db.pd_dataframe["echo"] == 1]["t2"].values)
    b1_vals = torch.from_numpy(db.pd_dataframe[db.pd_dataframe["echo"] == 1]["b1"].values)
    # we want to use the torch ADAM optimizer to optimize our function exploiting torchs internal tools
    # implement slice wise
    shape = data_nii.shape[:3]
    t2 = torch.zeros(shape)
    r2p = torch.zeros(shape)
    b1 = torch.zeros(shape)
    err_l2 = torch.zeros(shape)
    rmse_b1 = torch.zeros(shape)
    # t2_init = torch.zeros(shape)
    # b1_init = torch.zeros(shape)
    for idx_slice in range(data_nii.shape[2]):
        log_module.info(f"Process slice {idx_slice + 1} of {data_nii.shape[2]}")
        # # try autograd model
        # # set up model for slice
        # slice_optimize_model = DictionaryMatchingTv(
        #     db_torch_mag=db_torch_mag, db_t2s_s=t2_vals, db_b1s=b1_vals, slice_signal=data_nii[:, :, idx_slice],
        #     delta_t_r2p_ms=delta_t_ms_to_se, device=torch.device("cuda:0"),
        #     t2_range_s=(torch.min(t2_vals), torch.max(t2_vals)), autograd=True,
        #     b1_range=(torch.min(b1_vals), torch.max(b1_vals)), r2p_range_Hz=(0.01, 200)
        # )
        # # Instantiate optimizer
        # # torch autograd
        # # opt = torch.optim.Adam(slice_optimize_model.parameters(), lr=0.02)
        # opt = torch.optim.SGD(slice_optimize_model.parameters(), lr=0.02, momentum=0.9)
        # losses = optimization(model=slice_optimize_model, optimizer=opt, n=500)

        # brute force
        slice_optimize_model = BruteForce(
            db_torch_mag=db_torch_mag, db_t2s_s=t2_vals, db_b1s=b1_vals, slice_signal=data_nii[:, :, idx_slice],
            delta_t_r2p_ms=delta_t_ms_to_se, device=torch.device("cuda:0"),
            r2p_range_Hz=(0.1, 60), r2p_sampling_size=250, slice_b1=b1_nii[:, :, idx_slice]
        )
        slice_optimize_model.estimate_values()

        # yabox
        # yb_best = yb_optimization(slice_optimize_model)
        # this is a 3 * nx * ny vector bound between 0, 1, can use the defined model to translate this to the maps
        # t2_t2p_b1 = torch.reshape(
        #     torch.from_numpy(yb_best[0]),
        #     (3, slice_optimize_model.nx, slice_optimize_model.ny)
        # )
        # slice_optimize_model.set_parameters(t2_t2p_b1=t2_t2p_b1)

        # get slice maps
        (t2[:, :, idx_slice], r2p[:, :, idx_slice], b1[:, :, idx_slice],
         err_l2[:, :, idx_slice], rmse_b1[:, :, idx_slice]) = slice_optimize_model.get_maps()
        save_path = plib.Path(fit_config.save_path)
        # plot_loss(losses, save_path=save_path, title=f'losses_slice_{idx_slice + 1}')
        # plot_maps(t2, r2p, b1, t2_init, b1_init, save_path=save_path, title=f"maps_slice{idx_slice + 1}")
    return t2.numpy(), r2p.numpy(), b1.numpy(), err_l2.numpy(), rmse_b1.numpy()


def mese_fit(
        fit_config: options.FitConfig, data_nii: torch.tensor, name: str,
        db_torch_mag: torch.tensor, db: DB, device: torch.device, b1_nii=None):
    # get scaling of signal as one factor to compute pd
    rho_s = torch.linalg.norm(data_nii, dim=-1, keepdim=True)
    # for a first approximation we fit only the spin echoes
    # make 2d [xyz, t]
    data_nii_input_shape = data_nii.shape
    data_nii = torch.reshape(data_nii, (-1, data_nii.shape[-1]))
    # l2 normalize data - db is normalized
    data_nii = torch.nan_to_num(
        torch.divide(data_nii, torch.linalg.norm(data_nii, dim=-1, keepdim=True)),
        nan=0.0, posinf=0.0
    ).to(device)
    # nii_data = torch.reshape(data_nii_se, (-1, data_nii_input_shape[-1])).to(device)
    num_flat_dim = data_nii.shape[0]
    db_torch_mag = db_torch_mag.to(device)
    # get emc simulation norm
    # db_torch_norm = torch.squeeze(torch.from_numpy(db_norm)).to(device)
    db_torch_norm = torch.linalg.norm(db_torch_mag, dim=-1, keepdim=True)
    db_torch_mag = torch.nan_to_num(
        torch.divide(db_torch_mag, db_torch_norm), posinf=0.0, nan=0.0
    )
    db_torch_norm = torch.squeeze(db_torch_norm)

    batch_size = 3000
    nii_idxs = torch.split(torch.arange(num_flat_dim), batch_size)
    nii_zero = torch.sum(torch.abs(data_nii), dim=-1) < 1e-6
    nii_data = torch.split(data_nii, batch_size, dim=0)
    # make scaling map
    t2_vals = torch.from_numpy(db.pd_dataframe[db.pd_dataframe["echo"] == 1]["t2"].values).to(device)
    b1_vals = torch.from_numpy(db.pd_dataframe[db.pd_dataframe["echo"] == 1]["b1"].values).to(device)

    # b1 penalty of b1 database vs b1 input
    # dim b1 [xyz], db b1 - values for each entry [t1 t2 b1]
    if b1_nii is not None:
        b1_scale = fit_config.b1_tx_scale
        b1_nii = torch.reshape(b1_nii, (num_flat_dim,)).to(device)
        if torch.max(b1_nii) > 10:
            b1_nii = b1_nii / 100
        b1_nii *= b1_scale
        b1_nii = torch.split(b1_nii, batch_size, dim=0)
        use_b1 = True
        b1_weight = fit_config.b1_weight
        name = f"{name}_b1-in-w-{b1_weight}".replace(".", "p")
        if abs(1.0 - b1_scale) > 1e-3:
            name = f"{name}_tx-scale-{b1_scale:.2f}"
    else:
        use_b1 = False
        b1_weight = 0.0
        b1_scale = 1.0

    t2 = torch.zeros(num_flat_dim, dtype=t2_vals.dtype, device=device)
    l2 = torch.zeros(num_flat_dim, dtype=t2_vals.dtype, device=device)
    b1 = torch.zeros(num_flat_dim, dtype=b1_vals.dtype, device=device)
    # get scaling of db curves as one factor to compute pd
    rho_theta = torch.zeros(num_flat_dim, dtype=rho_s.dtype, device=device)

    # need to bin data for memory reasons
    for idx in tqdm.trange(len(nii_data), desc="batch processing"):
        data_batch = nii_data[idx]

        # l2 norm difference of magnitude data vs magnitude database
        # calculate difference, dims db [t2s t1 t2 b1, t], nii-batch [x*y*z*,t]
        l2_norm_diff = torch.linalg.vector_norm(db_torch_mag[:, None] - data_batch[None, :], dim=-1)

        # b1 penalty
        if use_b1:
            b1_batch = b1_scale * b1_nii[idx]
            b1_penalty = torch.sqrt(torch.square(b1_vals[:, None] - b1_batch[None, :]))
        else:
            b1_penalty = 0.0

        evaluation_matrix = b1_weight * b1_penalty + (1.0 - b1_weight) * l2_norm_diff

        # find minimum index in db dim
        min_idx = torch.argmin(evaluation_matrix, dim=0)
        # find minimum l2
        min_l2 = torch.min(l2_norm_diff, dim=0).values
        # populate maps
        t2[nii_idxs[idx]] = t2_vals[min_idx]
        b1[nii_idxs[idx]] = b1_vals[min_idx]
        l2[nii_idxs[idx]] = min_l2
        rho_theta[nii_idxs[idx]] = db_torch_norm[min_idx]
    # set t2 0 for signal 0 (eg. for bet) We could in principle use this to reduce computation demands
    # by not needing to compute those entries,
    # however we want to estimate the b1
    t2[nii_zero] = 0.0
    l2[nii_zero] = 0.0

    # reshape
    if torch.max(t2) < 5:
        # cast to ms
        t2 = 1e3 * t2
    t2 = torch.reshape(t2, data_nii_input_shape[:-1])
    t2 = t2.numpy(force=True)
    l2 = torch.reshape(l2, data_nii_input_shape[:-1])
    l2 = l2.numpy(force=True)
    b1 = torch.reshape(b1, data_nii_input_shape[:-1]).numpy(force=True)
    rho_theta[nii_zero] = 0.0
    rho_theta = torch.reshape(rho_theta, data_nii_input_shape[:-1]).cpu()

    pd = torch.nan_to_num(
        torch.divide(
            torch.squeeze(rho_s),
            torch.squeeze(rho_theta)
        ),
        nan=0.0, posinf=0.0
    )
    # we want to calculate histograms for both, and find upper cutoffs of the data values based on the histograms
    # since both might explode
    pd_hist, pd_bins = torch.histogram(pd.flatten(), bins=200)
    # find percentage where 95 % of data lie
    pd_hist_perc = torch.cumsum(pd_hist, dim=0) / torch.sum(pd_hist, dim=0)
    pd_cutoff_value = pd_bins[torch.nonzero(pd_hist_perc > 0.95)[0].item()]
    pd = torch.clamp(pd, min=0.0, max=pd_cutoff_value).numpy(force=True)
    return t2, b1, pd, l2


def main():



    # set path
    path = plib.Path(fit_config.save_path).absolute()
    log_module.info(f"setup save path: {path.as_posix()}")
    path.mkdir(parents=True, exist_ok=True)

    # load in data
    if fit_config.save_name_prefix:
        fit_config.save_name_prefix += f"_"
    in_path = plib.Path(fit_config.nii_path).absolute()
    stem = in_path.stem
    for suffix in in_path.suffixes:
        stem = stem.removesuffix(suffix)
    name = f"{fit_config.save_name_prefix}{stem}"

    log_module.info("__ Load data")
    data_nii, db, b1_nii, data_affine, b1_affine = io.load_data(fit_config=fit_config)
    data_nii_input_shape = data_nii.shape

    # for now take only magnitude data
    db_mag, db_phase, db_norm = db.get_numpy_array_ids_t()
    # device
    device = torch.device("cuda:0")

    # set echo types
    if not fit_config.echo_props_path:
        warn = "no echo properties given, assuming SE type echo train."
        log_module.warning(warn)
        t2, b1, pd, err_l2 = mese_fit(
            fit_config=fit_config, data_nii=data_nii, name=name, db_torch_mag=torch.from_numpy(db_mag),
            db=db, device=device, b1_nii=b1_nii
        )
        r2p = None
        rmse_b1 = None
    else:
        t2, r2p, b1, err_l2, rmse_b1 = megesse_fit(
            fit_config=fit_config, data_nii=data_nii, db_torch_mag=torch.from_numpy(db_mag),
            db=db, name=name, b1_nii=b1_nii
        )
        pd = None
    # calculate r2
    r2 = np.nan_to_num(1000.0 / t2, nan=0.0, posinf=0.0)

    # save
    names = [f"t2", f"r2", f"b1",]
    data = [t2, r2, b1]
    if pd is not None:
        data.append(pd)
        names.append("pd_like")
    if err_l2 is not None:
        data.append(err_l2)
        names.append("err_l2")
    if rmse_b1 is not None:
        data.append(rmse_b1)
        names.append("rmse_b1")
    if r2p is not None:
        data.append(r2p)
        names.append("r2p")
        r2s = r2 + r2p
        data.append(r2s)
        names.append("r2s")
    for idx in range(len(data)):
        save_nii_data(data=data[idx], affine=data_affine, name_prefix=name, name=names[idx], path=path)


if __name__ == '__main__':
    main()