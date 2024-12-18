import logging
import tqdm
import numpy as np
import torch
import collections
log_module = logging.getLogger(__name__)


def cgd(
        func_operator, x:torch.Tensor, b: torch.Tensor,
        iter_bar: tqdm.tqdm = None, max_num_iter: int = None, conv_tol: float = 1e-3):
    """
    The following Python program implements a Conjugate Gradient Descent (CGD)
    algorithm to solve a linear system of the form (Ax = b)
    :param func_operator: Function operator that applies the linear system operator A to a tensor x.
    :param x: Input tensor representing the initial guess for the solution.
    :param b: Tensor representing the right-hand side of the linear system Ax = b.
    :param iter_bar: Optional tqdm progress bar for tracking iterations.
    :param max_num_iter: Maximum number of iterations for the conjugate gradient descent algorithm.
    :param conv_tol: Convergence tolerance for the residual norm.
    :return: A tuple containing the solution tensor, residuals vector, and a dictionary with minimum residual norm and iteration number.
    """
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
        if iter_bar is not None:
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


def randomized_svd(
        matrix: torch.Tensor, rank: int,
        power_projections: int = 0, oversampling: int = None):
    """
    Function calculates a randomized Singular Value Decomposition (SVD) for a given matrix.
    The algorithm uses random projections and, if necessary, power iterations to improve the accuracy of the approximation
    :param matrix: Input tensor containing the data matrix to be decomposed. The tensor may have additional batch dimensions.
    :param rank: Number of singular values and vectors to compute.
    :param power_projections: Number of power iterations to perform to improve the approximation quality.
    :param oversampling: Additional random vectors to sample to improve the robustness of the decomposition.
    :return: A tuple containing the left singular vectors (U), singular values (S), and right singular vectors (VH) of the input matrix.
    """
    # get matrix shape - ignore batch dims
    m, n = matrix.shape[-2:]
    # get batch dims
    num_batch_dims = len(matrix.shape) - 2
    l = 2 * rank if oversampling is None else rank + oversampling

    # want short side to be at front to sample across long side
    if m > n:
        matrix = torch.movedim(matrix, -2, -1)
        transpose = True
        m, n = n, m
    else:
        transpose = False

    # Generate a random Gaussian matrix
    sample_projection = torch.randn(
        (n, l),
        dtype=matrix.dtype, device=matrix.device
    )

    # Form the random projection, dim: [n, sampling size]
    sample_matrix = torch.matmul(matrix, sample_projection)

    for _ in range(power_projections):
        sample_matrix = torch.matmul(matrix, torch.matmul(matrix.T, sample_matrix))

    # Orthonormalize basis using QR decomposition
    q, _ = torch.linalg.qr(sample_matrix)

    # Obtain the low-rank approximation of the original matrix - project original matrix onto that orthonormal basis
    lr = torch.matmul(q.T, matrix)

    # Perform SVD on the low-rank approximation
    u, s, vh = torch.linalg.svd(lr, full_matrices=False)

    # s, vh should be approximately the matrix s, vh of the svd from random matrix theory
    # we can get the left singular values by back projection
    u_matrix = torch.matmul(q, u[..., :, :rank])
    vh = vh[..., :rank, :]

    if transpose:
        v_temp = vh
        vh = torch.movedim(u_matrix, -1, -2)
        u_matrix = torch.movedim(v_temp, -1, -2)

    return u_matrix, s[..., :rank], vh


def subspace_orbit_randomized_svd(matrix: torch.Tensor, rank: int, oversampling: int = None):
    # adopted from Kaloorazi and Lamare (2018): 10.1109/TSP.2018.2853137
    # output rank approximation
    # get shape
    m, n = matrix.shape[-2:]
    l = 2 * rank if oversampling is None else rank + oversampling

    # want short side to be at back to sample across long side
    if m < n:
        matrix = torch.movedim(matrix, -2, -1)
        transpose = True
        m, n = n, m
    else:
        transpose = False

    # 1) draw random matrices phi 1 and 2
    omega = torch.randn(
        (n, l),
        dtype=matrix.dtype, device=matrix.device
    )
    # 2) compute t1 and t2
    t1 = torch.matmul(matrix, omega)
    t2 = torch.matmul(matrix.mT, t1)
    # 3) compute qr decompositions
    q1, _ = torch.linalg.qr(t1)
    q2, _ = torch.linalg.qr(t2)
    # 4) compute m
    m = torch.matmul(
        q1.mT, torch.matmul(matrix, q2)
    )
    u_k, s_k, v_k = torch.linalg.svd(m, full_matrices=False)
    u_k = u_k[:, :rank]
    s_k = s_k[:rank]
    v_k = v_k[:rank]
    a_1 = torch.matmul(q1, u_k)
    a_2 = torch.matmul(q2, v_k.conj().T).mT
    # a = torch.matmul(
    #     torch.matmul(a_1, torch.diag_embed(s_k).to(a_1.dtype)),
    #     a_2
    # )
    if transpose:
        v_temp = a_2
        vh = torch.movedim(s_k, -1, -2)
        u_matrix = torch.movedim(a_1, -1, -2)
    return a_1, s_k, a_2

def subspace_orbit_randomized_svd_PS(matrix: torch.Tensor, rank: int, oversampling: int = None, device=None) -> tuple[torch.Tensor, ...]:
    """
    PyTorch implementation of Kaloorazi and Lamare (2018, DOI:10.1109/TSP.2018.2853137) that computes
    a low-rank approximation.
    :param matrix: Input matrix of dimension 2 with shape (m, n) where m >= n
    :param rank: Rank to approximate. Must be < n
    :return: Decomposed matrix parts (a1, s, a2) where (a1 * s) @ a2 is the low-rank approximation
    """
    torch._assert(matrix.ndim == 2, "matrix must be 2D")
    m, n = matrix.shape[-2:]
    torch._assert(m >= n, "matrix must have at least as many rows as columns")
    l = 2 * rank if oversampling is None else rank + oversampling

    # The following steps describe algorithm 4 of the referenced paper
    # 1) Draw a standard Gaussian matrix
    t2 = torch.randn(
        (n, l),
        dtype=matrix.dtype, device=matrix.device
    )
    t1 = torch.zeros((matrix.shape[0], t2.shape[1]), dtype=matrix.dtype, device=matrix.device)

    # 2, 3) compute t1 and t2
    t1 = torch.matmul(matrix, t2)
    t2 = torch.matmul(matrix.mT, t1)

    # 4) compute qr decompositions
    q1, _ = torch.linalg.qr(t1)
    q2, _ = torch.linalg.qr(t2)

    # 5) compute m
    m = torch.matmul(
        q1.mT, torch.matmul(matrix, q2)
    )

    # 6) Calculate rank truncated SVD
    u_k, s_k, v_k = torch.linalg.svd(m, full_matrices=False)
    u_k = u_k[:, :rank].contiguous()
    s_k = s_k[:rank]
    v_k = v_k[:rank, :].contiguous()

    # 7) Form the SOR-SVD-based low-rank approximation
    a_1 = torch.matmul(q1, u_k)
    a_2 = torch.matmul(q2, v_k.H).H
    return a_1, s_k, a_2

class DE:
    """ Differential evolution """
    def __init__(self, param_dim: int, data_dim: int, population_size: int = 10,
                 p_crossover: float = 0.9, differential_weight: float = 0.8,
                 max_num_iter: int = 1000, conv_tol: float = 1e-4,
                 device: torch.device = torch.get_default_device()):
        # set parameter dimensions and bounds
        self.param_dim: int = param_dim
        self.data_dim: int = data_dim

        # algorithm vars
        self.device: torch.device = device

        self.population_size: int = population_size
        self.p_crossover: float = p_crossover
        self.differential_weight: float = differential_weight

        self.max_num_iter: int = max_num_iter
        self.conv_tol: float = conv_tol

        # functions
        self.func = NotImplemented

    def set_fitness_function(self, func):
        """ set function to calculate fitness of agents."""
        self.func = func

    def optimize(self):
        """ find minimum of fittness function. """
        # initialize agents -> parents und 3 random selected per parent a, b and c
        # batch everything, assume dims [b, num_params]
        agents = torch.rand(
            (self.data_dim, self.population_size, self.param_dim),
            device=self.device
        )
        ba = []

        # calculate the fitness / loss per agent -> agents should find optimal params
        fitness_agents = self.func(agents)

        # get max loss within population to calculate convergence later
        # last_max_loss = torch.max(fitness_agents, dim=-1).values

        # start iteration per batch
        conv_counter = 0
        last_conv_idx = 0

        # get batch, push to device
        bar = tqdm.trange(self.max_num_iter, desc="DE optimization")
        update_bar = max(int(self.max_num_iter / 20), 1)
        for idx in bar:
            a, b, c = torch.rand(
                (3, self.data_dim, self.population_size, self.param_dim),
                device=self.device
            )

            # create random numbers and indices for each dims
            r_p = torch.rand(
                (self.data_dim, self.population_size, self.param_dim),
                device=self.device
            )
            r_index = torch.randint(
                low=0, high=self.param_dim, size=(self.data_dim, self.population_size),
                device=self.device
            ).unsqueeze(-1)

            # select components to mutate
            mask_crossover = r_p < self.p_crossover
            mask_indices = torch.arange(self.param_dim, device=self.device)[None, None] == r_index
            mutation_condition = mask_crossover | mask_indices
            # calculate new candidates for the condition
            y = torch.where(
                condition=mutation_condition,
                input=a + self.differential_weight * (b - c),
                other=agents
            )

            # calculate fitness of new candidates
            fitness_y = self.func(y)

            # check for improvement and update
            better_fitness = fitness_y < fitness_agents
            # update agents
            agents = torch.where(
                condition=better_fitness.unsqueeze(-1).to(device=self.device),
                input=y,
                other=agents
            )
            # update fitness
            fitness_agents = torch.where(
                condition=better_fitness,
                input=fitness_y,
                other=fitness_agents
            )

            # get best agents within population
            agents_min = torch.min(fitness_agents, dim=-1)
            best_agent = agents[torch.arange(self.data_dim), agents_min.indices]
            # calculate convergence as max difference between best agent to last iteration.
            convergence = torch.linalg.norm(torch.max(fitness_agents, dim=-1).values)
            # ToDo: think about reducing the number of agents to process based on convergence criterion.
            # i.e. exclude converged agents from future iterations

            ba.append(best_agent)
            if convergence < self.conv_tol:
                if conv_counter > 10 and last_conv_idx == idx - 1:
                    bar.postfix = f"converged at iteration: {idx} :: conv: {convergence:.6f}"
                    break
                last_conv_idx = idx
                conv_counter += 1
            if idx % update_bar == 0:
                bar.postfix = f"convergence: {convergence:.6f}"
        return best_agent, ba


