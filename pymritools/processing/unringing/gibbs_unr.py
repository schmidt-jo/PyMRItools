"""
Gibbs unringing algorithm based on doi/10.1002/mrm.26054
Kellner et al. 2015 MRM
_____
Implementation, Jochen Schmidt, 26.09.2024

"""
import pathlib as plib
import nibabel as nib
from .functions import gibbs_unring_nd
import logging
log_module = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    # prepare working path
    path = plib.Path("/data/pt_np-jschmidt/code/gibun/dev")
    path.mkdir(exist_ok=True, parents=True)

    # load in data
    path_to_phantom_data = plib.Path(
        "/data/pt_np-jschmidt/data/00_phantom_scan_data/semc_2024-04-19/processed/semc/semc_r0p7_echo_mag.nii.gz"
    )
    log_module.info(f"load file: {path_to_phantom_data}")
    phantom_img = nib.load(path_to_phantom_data)
    phantom_data = phantom_img.get_fdata()

    phantom_unring = gibbs_unring_nd(image_data_nd=phantom_data, visualize=False, gpu=True)

    # save data
    path_to_save = path_to_phantom_data.with_stem(f"ur_{path_to_phantom_data.stem}").with_suffix(".nii")
    log_module.info(f"save file: {path_to_save}")
    img = nib.Nifti1Image(phantom_unring, phantom_img.affine)
    nib.save(img, path_to_save.as_posix())


if __name__ == '__main__':
    main()
