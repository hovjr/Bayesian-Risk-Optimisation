import math
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simpson
from THESIS.Code.numerical_methods import composite_simpsons, get_integration_points, monte_carlo
import time

import rpy2.robjects as robjects

r_functions = robjects.r('''source('/Users/alkoulous/Desktop/Bath - MSc/THESIS/Code/GSD.R')''')
r_MAM = robjects.globalenv['MAM']

class Trial:

    def __init__(self, name=None, theta_mu=None, xy_sigma=None,
                 phase_2_mu=None, phase_2_sigma=None,
                 phase_3_mu=None, phase_3_sigma=None,
                 delta_fut=None, delta_eff=None,
                 p2_alpha=0.05, p2_beta=0.2, p3_alpha=0.025, p3_beta=0.1,
                 market_value=None, sensitivity=0.7, general_setup_cost=None,
                 p2_sample=None, p2_subject_cost=None, p2_setup_cost=None,
                 p3_sample=None, p3_subject_cost=None, p3_setup_cost=None,
                 interim_cost=None, sfu='obf'):

        # Programme name
        self.name = name

        # Treatment sampling distribution - Carrying over from Phase II
        self.xy_sigma = xy_sigma
        self.theta_mu = theta_mu
        self.theta_sigma = math.sqrt(2 * (xy_sigma ** 2))

        # Treatment prior distribution
        self.phase_3_mu = phase_3_mu if phase_3_mu is not None else theta_mu
        self.phase_3_sigma = phase_3_sigma if phase_3_sigma is not None else math.sqrt(2 * math.pow(xy_sigma, 2))

        # Hypothesis testing conditions
        self.delta_eff = delta_eff if delta_eff is not None else theta_mu
        self.delta_fut = delta_fut if delta_fut is not None else 0
        self.p2_alpha = p2_alpha
        self.p2_beta = p2_beta
        self.p3_alpha = p3_alpha
        self.p3_beta = p3_beta
        self.assurance = None

        # Trial utility parameters
        self.market_value = market_value  # At target delta_eff
        self.sensitivity = sensitivity
        self.general_setup_cost = general_setup_cost

        self.p2_setup_cost = p2_setup_cost
        self.p2_subject_cost = p2_subject_cost
        self.p2_sample = p2_sample

        self.p3_setup_cost = p3_setup_cost
        self.p3_subject_cost = p3_subject_cost
        self.p3_sample = p3_sample

        # GSD design
        self.interim_cost = interim_cost
        self.sfu = sfu

        # Optimisation temp
        self.k_optimal = None
        self.total_utility = []
        self.expected_utility = []

    def suggest_N3(self):
        # Calculate required sample size in phase 3 to establish treatment effect delta_eff.
        z_alpha = norm.ppf(1 - self.p3_alpha)
        z_beta = norm.ppf(1 - self.p3_beta)

        sigma = self.phase_3_sigma
        # Estimated required sample size - Total of both treatments
        sample = 4 * math.pow(sigma, 2) * math.pow(((z_alpha + z_beta) / self.delta_eff), 2)

        # Balanced sample required
        n_xy = math.ceil(sample / 2)
        total_n = n_xy * 2

        # No interim expected sample size
        self.p3_sample = total_n

        return total_n, n_xy

    def suggest_N2(self):
        # Calculate required sample size in phase 3 to establish treatment effect delta_eff.
        z_alpha = norm.ppf(1 - self.p2_alpha)
        z_beta = norm.ppf(1 - self.p2_beta)

        sigma = self.xy_sigma
        # Estimated required sample size - Total of both treatments
        sample = 4 * math.pow(sigma, 2) * math.pow(((z_alpha + z_beta) / self.delta_eff), 2)

        # Balanced sample required
        n_xy = math.ceil(sample / 2)
        total_n = n_xy * 2

        # No interim expected sample size
        self.p2_sample = total_n

        return total_n, n_xy

    def power(self, theta, total_n):
        return norm.cdf((theta * math.sqrt(total_n) / (2 * self.xy_sigma)) + norm.ppf(self.p3_alpha))

    def compute_assurance(self, total_n, how='Simpson_manual', r=15, n_points=1000):
        # Computes Assurance or Probability of Success
        dist = norm(self.phase_3_mu, self.phase_3_sigma)
        x_list = get_integration_points(r=r, prior=dist)
        if how == 'Simpson_manual':

            gamma_theta = lambda theta: dist.pdf(theta) * self.power(theta, total_n)
            self.assurance = composite_simpsons(gamma_theta, x_list)

        elif how == 'Monte_Carlo':
            gamma_theta = lambda theta: self.power(theta, total_n)
            self.assurance = monte_carlo(self, gamma_theta, n_points)
        return self.assurance

    def zeta_theta(self, theta_bar):
        # Accepts maximum multiplier, sensitivity of sigmoid, true and target theta, to produce
        # multiplier of market value depending on true effect of proposed treatment.
        # Sigmoid centered around 1 when the true treatment equals the target delta_eff.
        # 0 -> 2 times
        return 2 / (1 + np.exp(-1 * self.sensitivity * (theta_bar - self.delta_eff)))

    def utility(self, k, N3, n_sims=1000):

        # Market value scaled by true treatment effect if trial is successful
        Gz_theta = self.market_value * self.zeta_theta(self.phase_3_mu)

        # Probability of success
        assurance = 0
        p_3eSS = 0
        if k == 1:
            p_3eSS = N3
            assurance = self.compute_assurance(N3, how='Simpson_manual')
        elif k > 1:
            temp_n_xy_k = math.ceil((N3 / k) / 2)

            # prior sigma?
            temp_mam = r_MAM(n_inter=k, alpha=self.p3_alpha, beta=self.p3_beta, std=self.phase_3_sigma,
                             interesting=self.delta_eff, uninteresting=self.delta_fut, sfu=self.sfu,
                             n_sims=n_sims, true_mu=self.phase_3_mu, true_std=self.phase_3_sigma,
                             iter_nxy=temp_n_xy_k)

            p_3eSS, assurance = temp_mam

        expected_revenue = Gz_theta * assurance

        # General setup cost
        general_setup = self.general_setup_cost

        # Phase 2 costs
        p2_total = self.p2_setup_cost + (self.p2_subject_cost * self.p2_sample)
        upto_p2_cost = p2_total + general_setup

        # Phase 3 costs
        interim_costs = k * self.interim_cost
        p3_total = self.p3_setup_cost + (self.p3_subject_cost * p_3eSS)

        total_cost = general_setup + p2_total + p3_total + interim_costs
        enpv = expected_revenue - total_cost

        return {'eNPV': enpv, 'p_3eSS': p_3eSS, 'assurance': assurance,
                'N2': self.p2_sample, 'upto_p2_cost': upto_p2_cost,
                'p3_cost': p3_total, 'total_cost': total_cost}

    def custom_opt_p3(self, k, n_sims=3000, l_mult=0.8, u_mult=3):

        util_df = pd.DataFrame()

        self.p3_sample = self.suggest_N3()[0]

        lower = math.floor(l_mult * self.p3_sample)
        upper = math.ceil(u_mult * self.p3_sample)

        n3_range = []
        if k == 1: n3_range = range(lower, upper)
        elif k > 1: n3_range = range(4, upper + 4, 4)

        mid_n3_id = len(n3_range) // 2
        lower_id = 0
        upper_id = len(n3_range)

        record = {}
        mid_npv = -1
        left_npv = 0
        right_npv = 0

        ticker = 0
        while max([mid_npv, left_npv, right_npv]) != mid_npv:
            ticker += 1

            # Calculate npv of midpoint index and neighbours
            left_n3 = n3_range[mid_n3_id - 1]
            try:
                left_npv = record[left_n3]
            except:
                temp_utility = self.utility(k=k, N3=left_n3, n_sims=n_sims)
                record[left_n3] = left_npv
                temp_info = {'Iteration': ticker, 'k': k, 'P3 Sample': left_n3, 'id': mid_n3_id - 1}
                temp_info.update(temp_utility)
                util_df = pd.concat([util_df, pd.DataFrame([temp_info])], ignore_index=True)

            mid_n3 = n3_range[mid_n3_id]
            try:
                mid_npv = record[mid_n3]
            except:
                temp_utility = self.utility(k=k, N3=mid_n3, n_sims=n_sims)
                record[mid_n3] = mid_npv
                temp_info = {'Iteration': ticker, 'k': k, 'P3 Sample': mid_n3, 'id': mid_n3_id}
                temp_info.update(temp_utility)
                util_df = pd.concat([util_df, pd.DataFrame([temp_info])], ignore_index=True)

            right_n3 = n3_range[mid_n3_id + 1]
            try:
                right_npv = record[right_n3]
            except:
                temp_utility = self.utility(k=k, N3=right_n3, n_sims=n_sims)
                record[right_n3] = right_npv
                temp_info = {'Iteration': ticker, 'k': k, 'P3 Sample': right_n3, 'id': mid_n3_id + 1}
                temp_info.update(temp_utility)
                util_df = pd.concat([util_df, pd.DataFrame([temp_info])], ignore_index=True)

            # Update midpoint towards increasing npv
            if left_npv > mid_npv:
                upper_id = mid_n3_id
                mid_n3_id = (lower_id + mid_n3_id) // 2

            elif right_npv > mid_npv:
                lower_id = mid_n3_id
                mid_n3_id = (upper_id + mid_n3_id) // 2

        # print(util_df)

        return util_df.sort_values(by='eNPV', ascending=False).reset_index().head(1).to_dict('records')[0]
