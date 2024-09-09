import numpy as np
from scipy.stats import uniform, norm, sem

def get_integration_points(prior, r):
    x_list = []
    # Lower tail
    for i in range(1, r):
        x_list.append(prior.mean() + prior.std() * (-3 - 4 * np.log(r / i)))

    # Around mean
    for i in range(r, 5 * r + 1):
        x_list.append(prior.mean() + prior.std() * (-3 + 3 * (i - r) / (2 * r)))

    # Upper tail
    for i in range(5 * r + 1, 6 * r):
        x_list.append(prior.mean() + prior.std() * (3 + 4 * np.log(r / (6 * r - i))))

    # Find midpoints - Composite Simpson's
    midpoints = []
    for idx in range(len(x_list) - 1):
        midpoints.append((x_list[idx] + x_list[idx + 1]) / 2)

    total_xi = sorted(x_list + midpoints)

    return total_xi


def composite_simpsons(gamma_theta, x_list):
    # Initialize sum with first and last terms
    d = 1 / 6
    summation = d * (gamma_theta(x_list[0]) + gamma_theta(x_list[-1]))

    for idx, x in enumerate(x_list[1:-1]):
        if (idx + 2) % 2 == 0:
            # Odd indices - Expression finds even % 2 == 0 because y_0 is already evaluated
            d = x_list[idx + 2] - x_list[idx]
            summation += (4 / 6) * d * gamma_theta(x)
        else:
            # Even indices
            summation += (2 / 6) * d * gamma_theta(x)

    return summation


def monte_carlo(t, gamma_theta, n_points):
    eval_int = 0
    r_sample = norm(t.phase_2_mu, t.phase_2_sigma).rvs(size=n_points)
    for x in r_sample: eval_int += gamma_theta(x)
    return 1 / n_points * eval_int, sem(r_sample)
