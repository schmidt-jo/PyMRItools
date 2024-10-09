import torch
import pathlib as plib
import tqdm
from pymritools.utils.phantom import SheppLogan
from pymritools.utils import fft
from pymritools.recon.loraks import operators
import plotly.graph_objects as go


def main():
    path_fig = plib.Path("./dev/figs")
    path_fig.mkdir(exist_ok=True, parents=True)

    size_x, size_y = 200, 200
    # import shepp logan
    max_val = 100
    sl_phantom = SheppLogan((size_x, size_y), as_torch_tensor=True) * max_val
    # convert to k-space
    sl_k = fft(sl_phantom, img_to_k=True, axes=(0, 1))
    # set up sampling pattern - keep central phase encodes and skip some outer ones
    sampling_mask = torch.zeros_like(sl_k, dtype=torch.int)
    # sampling_mask[:, torch.randint(low=0, high=size_y, size=(int(size_y/2),))] = 1
    sampling_mask[:, ::3] = 1
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
    sl_image_recon_us = torch.abs(fft(sl_undersampled_k, img_to_k=False, axes=(0, 1)))
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Heatmap(z=sl_image_recon_us.numpy())
    # )
    # fig.show()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    access_matrix, hankel_matrix = construct_hankel_access_matrix(torch.ones(20), 3)

    # setup operator
    loraks_c_op = operators.C(k_space_dims=(size_x, size_y, 1, 1), radius=3)
    # initialize c_matrix with undersampled k-space data
    c_matrix_init = loraks_c_op.operator(k_space=sl_undersampled_k[:, :, None, None]).to(device)
    # set c_matrix to be tensor to be optimized
    c_matrix = c_matrix_init.clone().requires_grad_(True).to(device)
    m_matrix = loraks_c_op.operator(torch.ones_like(sl_k)).to(device)

    # introduce rank reduction by multiplication of constant tensor
    # s_factor = torch.ones(loraks_c_op.nb_size, dtype=torch.float32).to(device)
    # s_factor[10:] = 0.0

    # want to optimize for the rank value too - normalized via sigmoid function to go from 0 to neighborhood size
    rank_cutoff = torch.tensor([0.2], device=device, requires_grad=True)
    s_factor = torch.linspace(10, -10, loraks_c_op.nb_size, device=device)

    # get scaling factor for back transformation i.e. adjoint operation
    scaling_factor = torch.reshape(loraks_c_op.p_star_p(torch.ones_like(sl_k)), (size_x, size_y))

    # set some optimization vars
    max_iter = 5000
    lambda_data_consistency = 0.8
    grad_step = 0.1

    losses = []
    conv = []
    conv_counter = 0
    bar = tqdm.trange(max_iter, desc="Optimization")
    for _ in bar:
        # start = time.time_ns()
        u, s, vt = torch.linalg.svd(c_matrix, full_matrices=False)
        # end = time.time_ns()
        # total = 1e-6 * (end - start)
        # print(f"total ms: {total:.2f}")
        rc = torch.nn.Sigmoid()(rank_cutoff)    # get cutoff between 0 and 1
        s = s * torch.nn.Sigmoid()(s_factor + 10 * rc)

        loss_1 = torch.sum(s)

        c_recon_loraks = torch.matmul(
            torch.matmul(u, torch.diag(s).to(u.dtype)),
            vt
        )
        # k_recon_loraks = torch.reshape(loraks_c_op.operator_adjoint(c_recon_loraks), (size_x, size_y)) / scaling_factor

        # loss_2 = torch.linalg.norm(k_recon_loraks * sampling_mask - sl_undersampled_k)
        loss_2 = torch.linalg.norm((c_recon_loraks - c_matrix_init) * m_matrix)

        loss = lambda_data_consistency * loss_2 + (1 - lambda_data_consistency) * loss_1
        loss.backward()
        losses.append(loss.item())

        # grad step
        with torch.no_grad():
            c_matrix -= c_matrix.grad * grad_step
            rank_cutoff -= rank_cutoff.grad * grad_step
        convergence = torch.sum(torch.abs(c_matrix.grad))
        c_matrix.grad.zero_()
        rank_cutoff.grad.zero_()

        conv.append(convergence.item())
        if convergence < 1e-4:
            conv_counter += 1
            if conv_counter > 5:
                break

        bar.postfix = (
            f"loss 1 : {loss_1.item():.2f} -- loss 2 {loss_2.item():.2f} -- total_loss: {loss.item():.2f} -- "
            f"convergence: {convergence.item():.2f}, rank: {rc.item() * 29:.2f}, "
        )

    k_recon_loraks = torch.nan_to_num(
        torch.reshape(
            loraks_c_op.operator_adjoint(c_matrix.detach().clone()).cpu(),
            (size_x, size_y)
        ) / scaling_factor,
        nan=0.0, posinf=0.0, neginf=0.0
    ).cpu()

    recon_image = fft(k_recon_loraks, img_to_k=False, axes=(0, 1))

    fig = go.Figure()
    fig.add_trace(go.Scattergl(y=losses, name="loss"))
    fig.add_trace(go.Scattergl(y=conv, name="convergence"))
    fig.update_layout(title="Loss")
    file_name = path_fig.joinpath("losses").with_suffix(".html")
    print(f"saving figure to {file_name}")
    fig.write_html(file_name.as_posix())

    print(s)

    plot_fig(recon_image, path_fig, "image_recon", k_space=False, max_val=max_val)
    plot_fig(sl_image_recon_us, path_fig, "image_orig_us_fft", k_space=False, max_val=max_val)
    plot_fig(recon_image - sl_image_recon_us, path_fig, "image difference", k_space=False, max_val=max_val)
    plot_fig(k_recon_loraks, path_fig, "k_recon", k_space=True)
    plot_fig(sl_undersampled_k, path_fig, "k_orig", k_space=True)


def plot_fig(data_2d, path, name, k_space: bool = False, max_val: float = 1.0):
    data_2d = torch.abs(data_2d)
    zmin = 0
    zmax = max_val
    if k_space:
        data_2d = torch.log(data_2d)
        zmin = -14
        zmax = 0
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=data_2d.numpy(),
            # zmin=zmin, zmax=zmax,
            colorscale="Magma")
    )
    fig.update_layout(title=name)
    file_name = path.joinpath(name).with_suffix(".html")
    print(f"saving figure to {file_name}")
    fig.write_html(file_name.as_posix())

if __name__ == '__main__':
    main()


