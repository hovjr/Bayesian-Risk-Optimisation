import math
import pandas as pd
import numpy as np
from scipy.stats import norm
from THESIS.Code.Trial import Trial
import time
path = '/Users/alkoulous/Desktop/Bath - MSc/THESIS/Thesis/'

def simulate_theta(t, n_xy, mu=None):
    mu = t.phase_2_mu if mu is None else mu
    xy_sigma = t.phase_2_sigma
    data = norm(mu, xy_sigma).rvs(n_xy)
    return np.mean(data)


def test_hypothesis(t, theta_bar, total_n):
    p_value = 1 - norm.cdf(theta_bar * math.sqrt(total_n) / (2 * t.xy_sigma))
    decision = p_value <= t.alpha
    return {'reject_H0': decision, 'p_value': p_value}


def simple_trial(theta_mu, xy_sigma, delta_eff, phase_2_mu=None, phase_2_sigma=None):
    # Initialize trial with specified treatment means; default alpha, power and delta_eff=x_mu-y_mu
    trial = Trial(theta_mu=theta_mu, xy_sigma=xy_sigma, delta_eff=delta_eff,
                  phase_2_mu=phase_2_mu, phase_2_sigma=phase_2_sigma)

    # Calculate sample size required in Phase 3. Default alpha, beta, delta_eff set to theta (mu_x-mu_y)
    total_n, n_xy = trial.suggest_N()

    # Simulate true theta values for a mean estimate of theta
    theta_bar = simulate_theta(trial, n_xy, mu=theta_mu)

    # Test the hypothesis using half the sample size (n/2 for each treatment, for n total)
    hypothesis_test = test_hypothesis(trial, theta_bar, total_n)

    # Compute assurance - According to prior defined in trial initialisation
    assurance = trial.compute_assurance(total_n, how='Simpson_manual', r=12)

    return {'Theta': theta_mu, 'xy_sigma': xy_sigma, 'delta_eff': delta_eff,
            'alpha': trial.alpha, 'beta': trial.beta, 'Sample size': total_n, 'Prior mu': phase_2_mu,
            'Prior sigma': phase_2_sigma, 'Assurance': assurance,
            'Simulated theta': round(theta_bar, 2), 'Hypothesis test': hypothesis_test['reject_H0'],
            'P value': hypothesis_test['p_value']}


# v0 = simple_trial(theta_mu=30, xy_sigma=45, delta_eff=20, phase_2_mu=25, phase_2_sigma=15)
# for key, value in v0.items(): print(key, value)

def simulate_power(effects, xy_sigma, delta_eff):
    # Simulate trials to estimate power over chosen number of iterations, for a fixed delta_eff and
    # varying treatment effect theta.
    start = time.process_time()

    results = pd.DataFrame()
    for effect in effects:
        temp_trial = Trial(theta_mu=effect, xy_sigma=xy_sigma, delta_eff=delta_eff)
        temp_total_n, temp_n_xy = temp_trial.suggest_N()
        for iterations in [500, 1500, 7500]:
            successful_trials = 0
            for i in range(iterations + 1):
                theta_bar = simulate_theta(temp_trial, temp_n_xy)
                if test_hypothesis(temp_trial, theta_bar, temp_total_n)['reject_H0']:
                    successful_trials += 1

            # Power estimated by the proportion of times the null hypothesis was rejected
            power_est = successful_trials / iterations

            # Bernoulli sample variance = pq, SE = sqrt(pq/sample_size)
            power_SE = math.sqrt(power_est * (1 - power_est) / iterations)

            # Theoretical power using the sampling distribution for Theta
            theoretical_power = temp_trial.power(temp_trial.phase_2_mu, temp_total_n)

            temp_row = {'\theta': ' ', 'Theoretical power': theoretical_power,
                        'm': iterations, 'Estimated power': power_est, 'Standard Error': power_SE}

            results = pd.concat([results, pd.DataFrame([temp_row])], ignore_index=True)

    print(time.process_time() - start)

    return results


# power_simulation = simulate_power(effects=[15, 20, 25], xy_sigma=45, delta_eff=20)
# power_simulation.to_latex(path + r'tables/power_simulation.tex',
#                           bold_rows=True, float_format="%1.3f", escape=True, index=False)


def assurances_table(mus, sigmas, theta_mu=30, xy_sigma=45, delta_eff=20):
    results = pd.DataFrame()

    for phase_2_mu in mus:
        for phase_2_sigma in sigmas:
            temp_trial = Trial(theta_mu=theta_mu, xy_sigma=xy_sigma, delta_eff=delta_eff,
                               phase_2_mu=phase_2_mu, phase_2_sigma=phase_2_sigma)

            total_n = temp_trial.suggest_N()[0]

            # start_simpsons = time.process_time()
            simp_assurance = temp_trial.compute_assurance(total_n, how='Simpson_manual', r=16)
            # total_simpsons_time = time.process_time() - start_simpsons

            # start_MC = time.process_time()
            # MC_assurance = temp_trial.compute_assurance(total_n, how='Monte_Carlo', n_points=1000)
            # total_MC_time = time.process_time() - start_MC

            temp_row = {'\theta': ' ', '\sigma': phase_2_sigma, 'Assurance': simp_assurance}

            results = pd.concat([results, pd.DataFrame([temp_row])], ignore_index=True)

    return results
# assurances = assurances_table(mus=[15, 30, 45], sigmas=[10, 30, 50], theta_mu=30, xy_sigma=45, delta_eff=20)
# assurances.to_latex(path + r'tables/assurances.tex',
#                           bold_rows=True, float_format="%1.3f", escape=True, index=False)

def MC_assurance(ns, phase_2_sigma, phase_2_mu, theta_mu, xy_sigma, delta_eff):
    results = pd.DataFrame()

    temp_trial = Trial(theta_mu=theta_mu, xy_sigma=xy_sigma, delta_eff=delta_eff,
                       phase_2_mu=phase_2_mu, phase_2_sigma=phase_2_sigma)

    total_n = temp_trial.suggest_N()[0]

    for N in ns:
        MC_assurance = temp_trial.compute_assurance(total_n, how='Monte_Carlo', n_points=N)
        temp_row = {'N': N, 'Assurance': MC_assurance[0], 'SE': MC_assurance[1]}
        results = pd.concat([results, pd.DataFrame([temp_row])], ignore_index=True)
    return results


# assurances_MC = MC_assurance(ns=[100, 200, 500, 1000, 3000, 10000, 50000, 100000, 200000],
#                              phase_2_mu=30, phase_2_sigma=30,
#                              theta_mu=30, xy_sigma=45, delta_eff=20)
# assurances_MC.to_latex(path + r'tables/assurances_MC.tex',
#                           bold_rows=True, float_format="%1.3f", escape=True, index=False)

def evaluate_p3_outcomes(t):
    # Demonstration of the various effects different posteriors have on the expected NPV of trials

    simulated_eNPV = pd.DataFrame()

    fixed_sample_p3 = t.suggest_N()[0]
    t.optimal_p3(k_max=12)

    t_info = {'Theta': t.theta_mu, 'XY sigma': t.xy_sigma, 'Theta sigma': t.theta_sigma,
              'delta_eff': t.delta_eff, 'alpha': t.alpha, 'beta': t.beta,
              'Sample size (P2)': t.p2_sample, 'Sample size (P3)': fixed_sample_p3}

    for L_mu in range(t.phase_2_mu - 2 * t.phase_2_sigma, t.phase_2_mu + 2 * t.phase_2_sigma + 1):
        # no_gs_assurance = t.compute_assurance(total_n=t_info['Sample size (P3)'])
        # expected_gain_gs =
        t.update_prior(theta_bar=8, known_variance=t.phase_2_sigma, n_xy=t.gs_exp_nxy)

        trial_utility = t.utility(p3_k_inter=t.k_optimal)

        temp_info = {'Posterior mean': t.phase_2_mu, 'Posterior sigma': t.phase_2_sigma,
                     'GS eNPV': trial_utility['eNPV']}

        simulated_eNPV = pd.concat([simulated_eNPV, pd.DataFrame([temp_info])], ignore_index=True)

    return t_info, simulated_eNPV

t0 = Trial(theta_mu=30, xy_sigma=45, delta_eff=20, delta_fut=0,
           market_value=1000000, general_setup_cost=80000,
           p2_sample=50, p2_subject_cost=200, p2_setup_cost=30000,
           p3_subject_cost=800, p3_setup_cost=50000, interim_cost=2000, sfu='obf')
# t0.custom_opt_p3(k=1)
# t0.utility(k=1, N3=149)


# util_df = t0.custom_opt_p3(k=1, l_mult=0.2, u_mult=3, n_sims=3000)
# util_df.filter(items=['iteration', 'P3 Sample', 'eNPV']).to_latex(path + r'tables/util_df.tex',
#                bold_rows=True, float_format="%1.0f", escape=True, index=False)

# test_programmes = ['H', 'E', 'I', 'M', 'P', 'R', 'F', 'O']
# demo_programmes = demo_programmes[demo_programmes['name'].isin(test_programmes)]