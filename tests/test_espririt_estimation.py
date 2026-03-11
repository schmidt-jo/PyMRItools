import pathlib as plib
import torch
import logging

from pymritools.modeling.espirit.functions import map_estimation
from pymritools.utils import torch_save, torch_load
from tests.utils import get_test_result_output_dir, ResultMode

def initial_tensor():
    # create complex tensor
    torch.random.manual_seed(1)
    a = torch.randn((25, 22, 20, 8), dtype=torch.complex64)
    # as k_rpsc
    espirit = map_estimation(
        k_rpsc=a, kernel_size=6, num_ac_lines=10, rank_fraction_ac_matrix=0.01, eigenvalue_cutoff=0.99
    )

    # get path
    path_out = plib.Path(get_test_result_output_dir("espirit_performance_dev", mode=ResultMode.TEST))
    torch_save(a, path_out, "initial_tensor")
    torch_save(espirit, path_out, "espirit_tensor")


def load_comp_tensors():
    path = plib.Path(get_test_result_output_dir("espirit_performance_dev", mode=ResultMode.TEST))
    tensor_in = torch_load(path.joinpath("initial_tensor").with_suffix(".pt"))
    tensor_espirit = torch_load(path.joinpath("espirit_tensor").with_suffix(".pt"))
    return tensor_in, tensor_espirit


def test_espirit_performance():
    tensor_in, tensor_espirit = load_comp_tensors()
    espirit = map_estimation(
        k_rpsc=tensor_in, kernel_size=6, num_ac_lines=10, rank_fraction_ac_matrix=0.01, eigenvalue_cutoff=0.99
    )

    assert torch.allclose(espirit, tensor_espirit)

def test_espririt_scaling():
    # create complex tensor
    torch.random.manual_seed(1)
    a = torch.randn((50, 50, 30, 16), dtype=torch.complex64)
    # as k_rpsc
    espirit = map_estimation(
        k_rpsc=a, kernel_size=6, num_ac_lines=10, rank_fraction_ac_matrix=0.01, eigenvalue_cutoff=0.99
    )
