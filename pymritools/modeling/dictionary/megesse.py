import logging
import pathlib as plib

import torch

from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.emc import EmcFitSettings
from pymritools.config.database import DB
from pymritools.utils import normalize_data, nifti_load, nifti_save

log_module = logging.getLogger(__name__)


def fit(settings: EmcFitSettings):


    # allocate space, no t1 fit for now, dont put on gpu,
    # since we only process the non zero data on gpu and fill in later
    t2_unmasked = torch.zeros(total_data_size, dtype=t2_vals.dtype, device=torch.device("cpu"))
    l2_unmasked = torch.zeros(total_data_size, dtype=t2_vals.dtype, device=torch.device("cpu"))
    b1_unmasked = torch.zeros(total_data_size, dtype=b1_vals.dtype, device=torch.device("cpu"))
    rho_theta_unmasked = torch.zeros(total_data_size, dtype=rho_s.dtype, device=torch.device("cpu"))

    # mask
    in_data_masked = input_data[mask_nii_nonzero]
    batch_dim_size = in_data_masked.shape[0]
    if b1_in:
        in_b1_masked = b1_data[mask_nii_nonzero]

    # set batch processing
    batch_size = settings.batch_size
    num_batches = int(np.ceil(batch_dim_size / batch_size))

    t2 = torch.zeros(batch_dim_size, dtype=t2_vals.dtype, device=device)
    l2 = torch.zeros(batch_dim_size, dtype=t2_vals.dtype, device=device)
    b1 = torch.zeros(batch_dim_size, dtype=b1_vals.dtype, device=device)
    rho_theta = torch.zeros(batch_dim_size, dtype=rho_s.dtype, device=device)

    if not b1_in:
        fit_db = db_torch_mag.to(device)
    # batch process
    for idx_b in tqdm.trange(num_batches, desc="Batch Processing"):
        start = idx_b * batch_size
        end = min(start + batch_size, batch_dim_size)
        data_batch = in_data_masked[start:end].to(device)

        # b1 penalty
        if b1_in:
            b1_batch = in_b1_masked[start:end]
            b1_penalty = torch.sqrt(torch.square(b1_vals[:, None] - b1_batch[None, :]))
            b1_min_idxs = torch.min(b1_penalty, dim=1).indices
            fit_db = db_torch_mag[:, :, b1_min_idxs, None].to(device)

        # l2 norm difference of magnitude data vs magnitude database
        # calculate difference, dims db [t1s, t2s, b1s,t], nii-batch [x*y*z*,t]
        l2_norm_diff = torch.linalg.vector_norm(
            fit_db[:, None] - data_batch[None, :], dim=-1)

        # find minimum index in db dim
        min_l2 = torch.min(l2_norm_diff, dim=0)
        min_idx_l2 = min_l2.indices.to(torch.device("cpu"))
        min_vals_l2 = min_l2.values.to(torch.device("cpu"))

        # populate maps
        t2[start:end] = t1t2b1_vals[min_idx_l2, 1]
        b1[start:end] = t1t2b1_vals[min_idx_l2, 2]
        l2[start:end] = min_vals_l2
        rho_theta[start:end] = rho_db[min_idx_l2]
    # fill in the whole tensor
    t2_unmasked[mask_nii_nonzero] = t2.cpu()
    l2_unmasked[mask_nii_nonzero] = l2.cpu()
    b1_unmasked[mask_nii_nonzero] = b1.cpu()
    rho_theta_unmasked[mask_nii_nonzero] = rho_theta.cpu()

    # reshape
    if torch.max(t2_unmasked) < 5:
        # cast to ms
        t2_unmasked = 1e3 * t2_unmasked
    t2_unmasked = torch.reshape(t2_unmasked, data_shape[:-1])
    t2_unmasked = t2_unmasked.numpy(force=True)
    l2_unmasked = torch.reshape(l2_unmasked, data_shape[:-1])
    l2_unmasked = l2_unmasked.numpy(force=True)
    b1_unmasked = torch.reshape(b1_unmasked, data_shape[:-1]).numpy(force=True)
    rho_theta_unmasked = torch.reshape(rho_theta_unmasked, data_shape[:-1]).cpu()
    r2 = np.divide(1e3, t2_unmasked, where=t2_unmasked > 1e-3, out=np.zeros_like(t2_unmasked))
    pd = torch.nan_to_num(
        torch.divide(
            torch.squeeze(rho_s),
            torch.squeeze(rho_theta_unmasked)
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

    # save data
    nifti_save(data=r2, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}r2")
    nifti_save(data=t2_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}t2")
    nifti_save(data=b1_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}b1")
    nifti_save(data=l2_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}l2-err")
    nifti_save(data=rho_s, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}rho-s")
    nifti_save(data=rho_theta_unmasked, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}rho-theta")
    nifti_save(data=pd, img_aff=input_img, path_to_dir=path_out, file_name=f"{name}pd-approx")

def main():
    # Setup CLI Program
    setup_program_logging(name="EMC Dictionary Combined R2 / R2* Search", level=logging.INFO)
    # Setup parser
    parser, prog_args = setup_parser(
        prog_name="EMC Dictionary Combined R2 / R2* Search",
        dict_config_dataclasses={"settings": EmcFitSettings}
    )
    # Get settings
    settings = EmcFitSettings.from_cli(args=prog_args.settings, parser=parser)
    settings.display()

    try:
        fit(settings=settings)
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    main()
