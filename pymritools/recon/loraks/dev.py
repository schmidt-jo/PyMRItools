import torch
import logging

from matplotlib.pyplot import title
from torch.nn.functional import mse_loss
from pymritools.utils.phantom import SheppLogan
from pymritools.utils import fft
from pymritools.recon.loraks import operators
import time
import plotly.graph_objects as go


def main():
    size_x, size_y = 200, 200
    # import shepp logan
    sl_phantom = SheppLogan((size_x, size_y), as_torch_tensor=True)
    # convert to k-space
    sl_k = fft(sl_phantom, inverse=True, axes=(0, 1))
    # set up sampling pattern - keep central phase encodes and skip some outer ones
    sampling_mask = torch.zeros_like(sl_k, dtype=torch.int)
    sampling_mask[:, torch.randint(low=0, high=size_y, size=(int(size_y/2),))] = 1
    sampling_mask[:, int(2/5 * size_y):int(3/5 * size_y)] = 1

    # fig = go.Figure()
    # fig.add_trace(
    #     go.Heatmap(z=torch.abs(sampling_mask).numpy())
    # )
    # fig.show()

    sl_undersampled_k = sl_k * sampling_mask

    # fig = go.Figure()
    # fig.add_trace(
    #     go.Heatmap(z=torch.log(torch.abs(sl_undersampled_k)).numpy())
    # )
    # fig.show()
    #
    sl_image_recon_us = torch.abs(fft(sl_undersampled_k, inverse=False, axes=(0, 1)))
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Heatmap(z=sl_image_recon_us.numpy())
    # )
    # fig.show()

    loraks_c_op = operators.C(k_space_dims=(size_x, size_y, 1, 1), radius=3)

    c_matrix_init = loraks_c_op.operator(k_space=sl_undersampled_k[:, :, None, None])

    max_iter = 200
    lambda_data_consistency = 0.9

    c_matrix = c_matrix_init.clone().requires_grad_(True)
    s_factor = torch.ones(29, dtype=torch.float32)
    s_factor[10:] = 0.0
    # c_matrix = torch.rand_like(c_matrix_init).requires_grad_(True)
    m_matrix = loraks_c_op.operator(k_space=sampling_mask[:, :, None, None])

    losses = []
    for _ in range(max_iter):
        # start = time.time_ns()
        u, s, vt = torch.linalg.svd(c_matrix, full_matrices=False)
        # end = time.time_ns()
        # total = 1e-6 * (end - start)
        # print(f"total ms: {total:.2f}")
        s = s * s_factor

        loss_1 = torch.sum(s)

        c_recon_loraks = torch.matmul(
            torch.matmul(u, torch.diag(s).to(u.dtype)),
            vt
        )

        loss_2 = torch.linalg.norm((c_recon_loraks - c_matrix_init) * m_matrix)
        loss = lambda_data_consistency * loss_2 + (1 - lambda_data_consistency) * loss_1
        loss.backward()

        print(f"loss 1: {loss_1.item():.2f}, loss 2: {loss_2.item():.2f}, total_loss: {loss.item():.2f} ")
        losses.append(loss.item())

        # grad step
        with torch.no_grad():
            c_matrix -= c_matrix.grad * 0.01
        c_matrix.grad.zero_()

    recon_k = torch.reshape(loraks_c_op.operator_adjoint(c_matrix.detach().clone()), (size_x, size_y))
    scaling_factor = torch.reshape(loraks_c_op.p_star_p(torch.ones_like(sl_k)), (size_x, size_y))
    recon_k = torch.nan_to_num(
        recon_k / scaling_factor,
        nan=0.0, posinf=0.0, neginf=0.0
    )

    recon_image = fft(recon_k, inverse=False, axes=(0, 1))

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(y=losses)
    )
    fig.update_layout(title="Loss")
    fig.show()
    print(s)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=torch.abs(recon_image).numpy())
    )
    fig.update_layout(title="Reconstruction")
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=torch.abs(sl_image_recon_us).numpy())
    )
    fig.update_layout(title="naive fft")
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=torch.abs(recon_image - sl_image_recon_us).numpy())
    )
    fig.update_layout(title="difference")
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(recon_k)).numpy())
    )
    fig.update_layout(title="Reconstruction")
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(sl_undersampled_k)).numpy())
    )
    fig.update_layout(title="naive fft")
    fig.show()




if __name__ == '__main__':
    main()


