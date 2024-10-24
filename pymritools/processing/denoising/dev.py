import logging
import pathlib as plib

import numpy as np
import torch

import plotly.colors as plc
import plotly.graph_objects as go
import plotly.subplots as psub
from scipy.linalg import eigvals

from twixtools import read_twix
from pymritools.config import setup_program_logging
from pymritools.utils import HidePrints
from pymritools.seqprog.rawdata.load_fns import get_whitening_matrix

log_module = logging.getLogger(__name__)


def distribution_mp(x, sigma, gamma):
    """
    Marchenko–Pastur distribution
    """
    lam_p = sigma**2 * (1 + np.sqrt(gamma))**2
    lam_m = sigma**2 * (1 - np.sqrt(gamma))**2
    result = np.zeros_like(x)
    mask = (lam_m < x) & (x < lam_p)
    result[mask] =  np.sqrt((lam_p - x[mask]) * (x[mask] - lam_m)) / (2 * np.pi * gamma * x[mask] * sigma**2)
    result /= np.sum(result)
    return result


def main():
    # set program logging
    setup_program_logging(name="MPPCA Denoising DEV", level=logging.INFO)
    # set up path to rd file
    path_to_file = plib.Path("./examples/raw_data/rd_file/meas_MID00030_FID05554_mese_acc_4_spoil_1800_rf_ad_z.dat").absolute()
    path_to_figs = plib.Path("./examples/raw_data/rd_file/figs").absolute()
    log_module.info(f"setup figure path: {path_to_figs.as_posix()}")
    path_to_figs.mkdir(exist_ok=True, parents=True)

    # read twix
    log_module.info(f"load twix file: {path_to_file.as_posix()}")
    with HidePrints():
        twix = read_twix(path_to_file.as_posix(), parse_geometry=True, verbose=True, include_scans=-1)[0]
    geometry = twix["geometry"]
    data_mdbs = twix["mdb"]
    hdr = twix["hdr"]

    # setup colors
    colors = plc.sample_colorscale("Turbo", np.linspace(0.1, 1, 7))

    log_module.debug(f"Remove SYNCDATA acquisitions and get Arrays".rjust(20))
    # remove data not needed, i.e. SYNCDATA acquisitions
    data_mdbs = [d for d in data_mdbs if "SYNCDATA" not in d.get_active_flags()]

    # get corresponding data
    data_noise = np.array([data_mdbs[k].data for k in range(2)])
    data = np.array([data_mdbs[k].data for k in np.random.randint(low=2, high=len(data_mdbs), size=(5,))])

    # pre-whiten
    psi_l_inv = get_whitening_matrix(noise_data_n_samples_channel=np.swapaxes(data_noise, -2, -1))
    data_noise = np.einsum("kmn, lm -> kln", data_noise, psi_l_inv, optimize=True)
    data = np.einsum("kmn, lm -> kln", data, psi_l_inv, optimize=True)

    # want a per channel estimation, move channels to front and make rest as square as possible
    data_noise = np.moveaxis(data_noise, 1, 0)
    # take same amount of samples as later
    data_noise = np.reshape(data_noise, (data_noise.shape[0], -1))
    # data_noise = data_noise[:, :data.shape[-1]]
    target = data_noise.shape[-1]
    a = int(np.ceil(np.sqrt(target)))
    for i in range(a):
        if target % a == 0:
            break
        else:
            a += 1
    # combine coils for now, we assume identical noise characteristics per coil (can do individual with more noise scans)
    data_noise = np.reshape(data_noise,(data_noise.shape[0], a, -1))
    n = np.max(data_noise.shape[-2:])
    # calculate line - based patched svds (all samples) - first dim = batch dim, number of noise scans in this case
    s = np.linalg.svdvals(data_noise)
    # get eigenvalues
    s_lam = s**2 / n
    cs_max = 1.2 * np.max(s_lam)
    hist_density = 25
    p_noise = np.zeros((data_noise.shape[0], hist_density))
    # fig = psub.make_subplots(
    #     rows=8, cols=4,
    #     shared_xaxes=True, shared_yaxes=True
    # )
    # for idx_c, c_svals in enumerate(s_lam):
        # histogramm per channel
    fig=go.Figure()
    bins_fixed = np.linspace(0, cs_max, hist_density+1)
    bin_mid_noise_est = bins_fixed[:-1] + np.diff(bins_fixed) / 2

    for idx_c, sc in enumerate(s_lam):
        hist, bins = np.histogram(sc, bins=bins_fixed, density=True)
        hist /= np.sum(hist)

        # get expectation value
        exp_mp = np.sum(hist * bin_mid_noise_est)

        sigma = np.sqrt(exp_mp)
        gamma = np.min(data_noise.shape[-2:]) / np.max(data_noise.shape[-2:])
        # get biggest and lowest s - sanity check
        sigma_2 = np.sqrt((np.max(sc) - np.min(sc)) / 4 / np.sqrt(gamma))
        # can take avg
        sigma = (sigma + sigma_2) / 2

        p_noise[idx_c] = distribution_mp(x=bin_mid_noise_est, sigma=sigma, gamma=gamma)
        p_corr = p_noise[idx_c][bin_mid_noise_est < 1.1 * sigma**2 * (1 + np.sqrt(gamma))**2]

        # first do a matched filtering on original noise data histogram
        corr = np.correlate(hist, p_corr, mode="full")
        corr_ax = np.concatenate((-np.flip(bin_mid_noise_est[1:corr.shape[0]-bin_mid_noise_est.shape[0]+1]), bin_mid_noise_est))
        # plot
        # row = idx_c // 4 + 1
        # col = idx_c % 4 + 1
        if idx_c == 0:
            fig.add_trace(
                go.Scattergl(y=0.2*(np.random.random(size=sc.shape)+0.5), x=sc, mode="markers",
                             name="s_vals", marker=dict(color=colors[0]))
                # row=row, col=col
            )
            fig.add_trace(
                go.Bar(x=bin_mid_noise_est, y=hist,
                        name="s_vals", marker=dict(color=colors[0]))
                # row=row, col=col
            )

            # fig.update_traces(orientation='h', side='positive', width=3, points=False)

            fig.add_trace(
                # go.Scattergl(x=bin_mid_noise_est, y=optim_func(bin_mid_noise_est, p_opt), name="s_vals")
                go.Scattergl(x=bin_mid_noise_est, y=p_noise[idx_c], name="p", mode="lines", marker=dict(color=colors[1])),
                # row=row, col=col
            )
            fig.add_trace(
                # go.Scattergl(x=bin_mid_noise_est, y=optim_func(bin_mid_noise_est, p_opt), name="s_vals")
                go.Scattergl(x=corr_ax, y=corr, name="hist correlated with p", mode="lines", marker=dict(color=colors[2])),
                # row=row, col=col
            )

    fig_name = path_to_figs.joinpath("noise_lines_singular_values.html")
    log_module.info(f"write file: {fig_name.as_posix()}")
    fig.write_html(fig_name.as_posix())

    # p is now our filter shape - the shape we are looking for in our data eigenvalues to match and identify noise
    # except with the matched filter usually one wants to detect the signal, we are interested in matching the noise
    # part of the eigenvalue spectrum
    # we want to build a weighting and down-weight singular values that we take to be belonging to above mp-distribution
    p_weight = 1 - p_noise / np.max(p_noise, axis=1, keepdims=True)
    p_weight_ax = bin_mid_noise_est

    # for each of the lines we want to do the same thing, first dim is batch
    # lets "square" the line, i.e. build a matrix with itself, dims [batch, channel, samples]
    a = int(np.ceil(np.sqrt(data.shape[-1])))
    for i in range(a):
        if data.shape[-1] % a == 0:
            break
        else:
            a += 1
    shape = data.shape
    data = np.reshape(data, (*shape[:2], a, -1))
    n = np.max(data.shape[-2:])
    # do svd
    u, s, v = np.linalg.svd(data, full_matrices=False)
    # get eigenvalues
    s_lam = s ** 2 / n

    # we dont actually need the histogram, we have a weighting factor p_weight based on
    # the value of the singular values (p_weight ax)
    # after computing the singular values all we need to do is to downweight them
    # based on the interpolated value of p_weight at their "position"

    # plot spectrum
    rows = int(np.min([10, s_lam.shape[0]]))
    fig = psub.make_subplots(rows=rows, cols=4)
    for idx, s_vals in enumerate(s_lam):
        # pick the specific line
        for idx_c, s_vals_c in enumerate(s_vals):
            if idx_c == 0 and idx == 0:
                showlegend = True
            else:
                showlegend = False
            # pick the channel
            # transfer p
            weight_at_s = np.interp(x=s_vals_c, xp=p_weight_ax, fp=p_weight[idx_c])
            if idx < rows:
                # histogramm for visualization
                bins = np.linspace(0, 1.2 * np.max(s_vals_c), 100)
                hist, bins = np.histogram(s_vals_c, bins=bins, density=True)
                hist /= np.sum(hist)
                bin_mid = bins[:-1] + np.diff(bins) / 2
                if idx_c == 0:
                    hist /= np.max(hist)
                    fig.add_trace(
                        go.Bar(x=bin_mid, y=hist, name="s_vals", marker=dict(color=colors[0]), showlegend=showlegend),
                        row=idx+1, col=1
                    )
                fig.add_trace(
                    go.Scattergl(x=s_vals_c, y=weight_at_s, name="weighting function",
                                 mode="lines", showlegend=showlegend),
                    row=idx + 1, col=1
                )
                # do correlation
                # fig.add_trace(
                #     go.Scattergl(x=bin_mid, y=corr, name="correlate",
                #                  mode="lines", marker=dict(color=colors[2]), showlegend=showlegend),
                #     row=idx + 1, col=1
                # )
            # compute weighting
            # scale correlation function to 1
            # corr_w = corr / np.max(corr)
            # weight = 1 - corr_w
            # # interpolate function at singular values
            # weight_at_s = np.interp(x=s_vals, xp=bin_mid, fp=weight)
            # weight singular vals
            s_w = s[idx, idx_c] * weight_at_s
            # reconstruct signal without filtered noise
            signal_filt = np.matmul(
                np.matmul(u[idx, idx_c], np.diag(s_w)), v[idx, idx_c]
            )
            # reshape to single line
            signal_filt = np.reshape(signal_filt, -1)
            # normalize to previous signal levels
            signal_filt /= np.max(np.abs(signal_filt)) / np.max(np.abs(data[idx, idx_c]))
            plot_data = data[idx, idx_c].flatten()
            # plot signals
            if idx < rows:
                if idx_c > 2:
                    break
                fig.add_trace(
                    go.Scattergl(y=np.abs(plot_data), name="noisy signal mag",
                                 mode="lines", marker=dict(color=colors[3]),
                                 showlegend=showlegend, legendgroup=0),
                    row=idx + 1, col=2 + idx_c
                )
                fig.add_trace(
                    go.Scattergl(y=np.angle(plot_data), name="noisy signal phase",
                                 mode="lines", marker=dict(color=colors[4]),
                                 showlegend=showlegend, legendgroup=1),
                    row=idx + 1, col=2 + idx_c
                )
                fig.add_trace(
                    go.Scattergl(y=np.abs(signal_filt), name="filtered signal mag",
                                 mode="lines", marker=dict(color=colors[5]),
                                 showlegend=showlegend, legendgroup=2),
                    row=idx + 1, col=2 + idx_c
                )
                fig.add_trace(
                    go.Scattergl(y=np.angle(signal_filt), name="filtered signal phase",
                                 mode="lines", marker=dict(color=colors[4]),
                                 showlegend=showlegend , legendgroup=3),
                    row=idx + 1, col=2 + idx_c
                )
    fig_name = path_to_figs.joinpath("pe_lines_singular_values.html")
    log_module.info(f"write file: {fig_name.as_posix()}")
    fig.write_html(fig_name.as_posix())



if __name__ == '__main__':
    main()
