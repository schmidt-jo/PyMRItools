import logging
import pickle
import json

import torch
import pathlib as plib

import plotly.graph_objects as go
import plotly.colors as plc
import plotly.subplots as psub

from pymritools.simulation.emc.sequence import MESE
from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.emc import EmcSimSettings, EmcParameters

from tests.utils import get_test_result_output_dir, ResultMode

logger = logging.getLogger(__name__)


def main():
    # set some paths
    path_out = plib.Path(
        get_test_result_output_dir("pulse_profile_visualisation", mode=ResultMode.VISUAL)
    )

    path_in_phantom_gauss = plib.Path(
        "C:\\DatenJo\\Daten\\02_work\\04_data\\proj_mese\\seq\\phantom\\"
        "20251206_mese_cfa130-rf26-38-gauss_rgs0p8_a3p5_trd40_m065-sl50-g67_read-t720-ge-m100\\pulse_profiles.pkl"
    )
    path_in_phantom_slr = plib.Path(
        "C:\\DatenJo\\Daten\\02_work\\04_data\\proj_mese\\seq\\phantom\\"
        "20251203_mese_cfa130-rf28-38-slr_rgs0p8_a3p5_trd20_m065-sl50-g67_read-t720-ge-m100\\pulse_profiles.pkl"
    )

    # load in profiles
    profiles_cfa_gauss = pickle.load(open(path_in_phantom_gauss, "rb"))
    profiles_cfa_slr = pickle.load(open(path_in_phantom_slr, "rb"))

    names = ["phantom_cfa_gauss", "phantom_cfa_slr"]

    for i, profile in enumerate([profiles_cfa_gauss, profiles_cfa_slr]):
        p_save = []
        for j, p in enumerate(profile):
            name = p["name"]
            if "post_pulse" in name:
                continue
            p_save.append({"name": name, "profile": p["profile"].tolist()})
        p_out = path_out.joinpath(names[i]).with_suffix(".json")
        with open(p_out, "w") as f:
            json.dump(p_save, f, indent=2)


if __name__ == '__main__':
    setup_program_logging("MESE pulse profile visualisation", logging.INFO)
    main()
