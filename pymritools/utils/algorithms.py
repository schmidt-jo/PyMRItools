import logging
import tqdm
import numpy as np
import torch
import collections
log_module = logging.getLogger(__name__)


def cgd(
        func_operator, x:torch.Tensor, b: torch.Tensor,
        iter_bar: tqdm.tqdm = None, max_num_iter: int = None, conv_tol: float = 1e-3):
    if iter_bar is not None:
        if max_num_iter is not None:
            iter_dict = collections.OrderedDict([("iter", "_".rjust(max_num_iter, "_"))])
            iter_bar.set_postfix(ordered_dict=iter_dict)

    n2b = torch.linalg.norm(b)

    x = torch.zeros_like(b)
    p = 1
    xmin = x
    iimin = 0
    tolb = conv_tol * n2b

    r = b - func_operator(x)

    normr = torch.linalg.norm(r)
    normr_act = normr

    if normr < tolb:
        log_module.info("convergence before loop")

    res_vec = torch.zeros(max_num_iter)
    normrmin = normr

    rho = 1

    for ii in range(max_num_iter):
        z = r
        rho1 = rho
        rho = torch.abs(torch.sum(r.conj() * r))
        if ii == 0:
            p = z
        else:
            beta = rho / rho1
            p = z + beta * p
        q = func_operator(p)
        pq = torch.sum(p.conj() * q)
        alpha = rho / pq

        x = x + alpha * p
        r = r - alpha * q

        normr = torch.linalg.norm(r)
        normr_act = normr
        res_vec[ii] = normr

        if normr <= tolb:
            normr_act = torch.linalg.norm(r)
            res_vec[ii] = normr_act
            log_module.info(f"reached convergence at step {ii + 1}")
            break

        if normr_act < normrmin:
            normrmin = normr_act
            xmin = x
            iimin = ii
            log_module.debug(f"min residual {normrmin:.2f}, at {iimin + 1}")
        iter_dict["iter"] = f"{iter_dict['iter'][1:]}I"
        iter_bar.set_postfix(iter_dict)
    return xmin, res_vec, {"norm_res_min": normrmin, "iteration": iimin}


def chambolle_pock_tv(data: np.ndarray, lam, n_it=100):
    """
    Chambolle-Pock algorithm for Total Variation regularization.

    The following objective function is minimized:
        ||K*x - d||_2^2 + Lambda*TV(x)
    Adapted from Pierre Paleo: https://github.com/pierrepaleo/spire
    Take operators K(x) as identity
    # TODO: Doc string doesn't fit provided parameters
    lam : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    L : norm of the operator [P, Lambda*grad] = 3.6 for identity
    n_it : number of iterations
    return_all: if True, an array containing the values of the objective function will be returned

    """

    sigma = 1.0 / 3.6
    tau = 1.0 / 3.6

    x = np.zeros_like(data)
    p = np.zeros_like(np.array(np.gradient(x)))
    q = np.zeros_like(data)
    x_tilde = np.zeros_like(x)

    for _ in range(0, n_it):
        # Update dual variables
        # For anisotropic TV, the prox is a projection onto the L2 unit ball.
        # For anisotropic TV, this is a projection onto the L-infinity unit ball.
        arg = p + sigma * np.array(np.gradient(x_tilde))
        p = np.minimum(np.abs(arg), lam) * np.sign(arg)
        q = (q + sigma * x_tilde - sigma * data) / (1.0 + sigma)
        # Update primal variables
        x_old = x
        x = x + tau * np.sum(p, axis=0) - tau * q
        x_tilde = x + (x - x_old)

    # constrain to >= 0
    x = np.clip(x, 0, np.max(x))
    return x
