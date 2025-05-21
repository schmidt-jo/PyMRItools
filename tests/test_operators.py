import torch
import pathlib as plib

import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.recon.loraks.operators import Operator, OperatorType
from tests.utils import get_test_result_output_dir, ResultMode


def create_test_tensor():
    # just visualizing k-space position and some random data in imaginary component
    nx, ny, nc = (20, 10, 2)
    x, y = torch.meshgrid(torch.arange(- nx // 2, nx // 2), torch.arange(-ny // 2, ny // 2), indexing="ij")

    k = x * y
    k = k[None] * torch.arange(1, nc+1)[:, None, None]
    k = k.to(torch.complex128) + torch.randn((nc, nx, ny), dtype=torch.complex128) * 1e-2
    return k


def figure_compare_k_matrix(k, k_rec, matrix, path, name):
    fig = psub.make_subplots(
        rows=3, cols=6,
        column_titles=["in c1", "out c1", "diff", "in c2", "out c2", "diff"],
        row_titles=["real", "imag", "matrix"],
        horizontal_spacing=0.02, vertical_spacing=0.05,
        specs=[
            [{}, {}, {}, {}, {}, {}],
            [{}, {}, {}, {}, {}, {}],
            [{"colspan": 3}, None, None, {"colspan": 3}, None, None],
        ]
    )
    rmax = torch.max(torch.abs(k)).item() * 0.8
    for c, d in enumerate([k, k_rec, k - k_rec]):
        for cn in range(2):
            p = d[cn]
            for r, f in enumerate([torch.real, torch.imag]):
                fig.add_trace(
                    go.Heatmap(
                        z=f(p).numpy() if torch.is_complex(p) else p.numpy(),
                        transpose=False, showscale=False,
                        colorscale="Inferno",
                        zmin=0 if r == 0 else None, zmax=rmax if r == 0 else None
                    ),
                    row=1 + r, col=1 + c + 3 * cn
                )
    if not matrix.is_complex():
        matrix = matrix + 1j * 1e-18
    for i, f in enumerate([torch.real, torch.imag]):
        fig.add_trace(
            go.Heatmap(
                z=f(matrix).numpy(),
                transpose=False, showscale=False,
                colorscale="Inferno"
            ),
            row=3, col=3 * i + 1
        )

    fn = path.joinpath(f"forward_adjoint_k_matrix_{name}").with_suffix(".html")
    fig.write_html(fn)


def process_operator_test(operator_type: OperatorType, device: torch.device):
    torch.manual_seed(0)
    k = create_test_tensor()

    op = Operator(
        k_space_shape=k.shape, nb_side_length=5, operator_type=operator_type, device=device
    )
    matrix = op.forward(k_space=k)

    k_recovered = op.adjoint(matrix)
    k_recovered /= op.count_matrix\

    print(f"")
    print(f"__{op.operator_type.name} Operator__")
    print(f"Forward - Adjoint, allclose: {torch.allclose(k, k_recovered)}")

    path = plib.Path(get_test_result_output_dir(f"{op.operator_type.name.lower()}_operator", mode=ResultMode.TEST)).absolute()

    figure_compare_k_matrix(k, k_recovered, matrix, path, op.operator_type.name)

    assert torch.allclose(k, k_recovered)


def test_s_operator():
    process_operator_test(OperatorType.S, torch.device("cpu"))


def test_c_operator():
    process_operator_test(OperatorType.C, torch.device("cpu"))
