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
        matrix: torch.Tensor, sampling_size: int,
        power_projections: int = 0, oversampling_factor: int = 0):
    """
    Function calculates a randomized Singular Value Decomposition (SVD) for a given matrix.
    The algorithm uses random projections and, if necessary, power iterations to improve the accuracy of the approximation
    :param matrix: Input tensor containing the data matrix to be decomposed. The tensor may have additional batch dimensions.
    :param sampling_size: Number of singular values and vectors to compute.
    :param power_projections: Number of power iterations to perform to improve the approximation quality.
    :param oversampling_factor: Additional random vectors to sample to improve the robustness of the decomposition.
    :return: A tuple containing the left singular vectors (U), singular values (S), and right singular vectors (VH) of the input matrix.
    """
    # get matrix shape - ignore batch dims
    m, n = matrix.shape[-2:]
    # get batch dims
    num_batch_dims = len(matrix.shape) - 2

    # want short side to be at front to sample across long side
    if m > n:
        matrix = torch.movedim(matrix, -2, -1)
        transpose = True
        m, n = n, m
    else:
        transpose = False

    # Generate a random Gaussian matrix
    sample_projection = torch.randn(
        (n, sampling_size + oversampling_factor),
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
    u_matrix = torch.matmul(q, u)

    if transpose:
        v_temp = vh
        vh = torch.movedim(u_matrix, -1, -2)
        u_matrix = torch.movedim(v_temp, -1, -2)

    return u_matrix, s, vh


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
        self.func_forward = None

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

        # calculate the fitness / loss per agent -> agents should find optimal params
        fitness_agents = self.func(agents)

        # get best agent within population to calculate convergence later
        agents_min = torch.min(fitness_agents, dim=-1)
        last_best_agent = agents[torch.arange(self.data_dim), agents_min.indices]

        # start iteration per batch
        conv_counter = 0
        last_conv_idx = 0

        # get batch, push to device
        bar = tqdm.trange(self.max_num_iter, desc="DE optimization")
        update_bar = int(self.max_num_iter / 20)
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
                low=0, high=self.data_dim, size=(self.data_dim, self.population_size),
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
            convergence = torch.max(torch.linalg.norm(best_agent - last_best_agent, dim=-1))
            last_best_agent = best_agent
            # ToDo: think about reducing the number of agents to process based on convergence criterion.
            # i.e. exclude converged agents from future iterations

            if convergence < self.conv_tol:
                if conv_counter > 10 and last_conv_idx == idx - 1:
                    bar.postfix = f"converged at iteration: {idx} :: conv: {convergence:.6f}"
                    break
                last_conv_idx = idx
                conv_counter += 1
            if idx % update_bar == 0:
                bar.postfix = f"convergence: {convergence:.5f}"
        return best_agent


