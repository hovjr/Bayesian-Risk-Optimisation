import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from THESIS.Code.Trial import Trial
path = '/Users/alkoulous/Desktop/Bath - MSc/THESIS/Thesis/'

def power_vs_theta(theta_mu=30, xy_sigma=45, delta_eff=20, theta_max=30):
    trial = Trial(theta_mu=theta_mu, xy_sigma=xy_sigma, delta_eff=delta_eff)

    theta_list = range(1, theta_max)
    pow_list = []
    for theta in theta_list:
        pow_list.append(trial.power(theta=theta, total_n=214))

    plt.plot(theta_list, pow_list)
    plt.show()

    df = pd.DataFrame({'Sample size': theta_list, 'Power': pow_list})

    df.to_csv(path + r'data/power_vs_theta.csv')

    return df
# power_vs_theta()

def power_vs_n(theta_mu=30, xy_sigma=45, delta_eff=20, n_max=500):
    trial = Trial(theta_mu=theta_mu, xy_sigma=xy_sigma, delta_eff=delta_eff)

    n_list = range(1, n_max)
    pow_list = []
    for n_3 in n_list:
        pow_list.append(trial.power(theta=20, total_n=n_3))

    plt.plot(n_list, pow_list)
    plt.show()

    df = pd.DataFrame({'Sample size': n_list, 'Power': pow_list})

    df.to_csv(path + r'data/power_vs_n.csv')

    return df
# power_vs_n()

def zeta_theta_demo(theta_mu=30, xy_sigma=45, delta_eff=20, sensitivies=[0.5, 1, 2]):

    df = pd.DataFrame()

    theta_list = np.linspace(10, 30, 200)
    for sens in sensitivies:
        trial = Trial(theta_mu=theta_mu, xy_sigma=xy_sigma, delta_eff=delta_eff, sensitivity=sens)
        for theta in theta_list:
            temp_row = {'Sensitivity': sens, 'theta': theta, 'zeta': trial.zeta_theta(theta)}
            df = pd.concat([df, pd.DataFrame([temp_row])], ignore_index=True)

    plt.plot(df['theta'], df['zeta'])
    plt.show()

    df.to_csv(path + r'data/zeta_plot.csv')

    return df
# df = zeta_theta_demo()

def eNPV_vary_S():

    df = pd.DataFrame()

    for S in [0.1, 0.2, 0.3, 0.5, 0.8, 1.5]:
        trial = Trial(theta_mu=30, xy_sigma=45,
                      phase_2_mu=30, phase_2_sigma=15,
                      delta_eff=25, delta_fut=0,
                      market_value=1000000, sensitivity=S,
                      general_setup_cost=80000,
                      p2_sample=50, p2_subject_cost=200, p2_setup_cost=30000,
                      p3_subject_cost=800, p3_setup_cost=50000, interim_cost=2000)

        for sample_size in range(1, 300, 1):
            temp_row = trial.utility(k=1, N3=sample_size, n_sims=3000)
            temp_row.update({'S': trial.sensitivity})
            df = pd.concat([df, pd.DataFrame([temp_row])], ignore_index=True)

        # plt.plot(df['p_3eSS'], df['eNPV'])
        # plt.show()
        print(S)

    # df.to_csv(path + r'data/eNPV_vary_S.csv')

    return df
# df = eNPV_vary_S()


def gradient_plot_methodol(csv_name='gradient_plot_1'):

    df = pd.DataFrame()

    trial = Trial(theta_mu=30, xy_sigma=45,
                  # phase_2_mu=30, phase_2_sigma=15,
                  delta_eff=20, delta_fut=0,
                  market_value=1000000, sensitivity=0.7,
                  general_setup_cost=80000,
                  p2_sample=50, p2_subject_cost=200, p2_setup_cost=30000,
                  p3_subject_cost=800, p3_setup_cost=50000, interim_cost=2000)


    custom_opt = trial.custom_opt_p3(k=1)
    for n3_max in range(custom_opt['P3 Sample'], 1, -4):
        temp_util = trial.utility(k=1, N3=n3_max, n_sims=1000)
        temp_row = {'Programme': trial.name,
                    'p2_mu': trial.theta_mu, 'p2_sigma': trial.xy_sigma,
                    'p3_mu': trial.phase_3_mu, 'p3_sigma': trial.phase_3_sigma,
                    'N_3_max': n3_max}
        temp_row.update(temp_util)
        df = pd.concat([df, pd.DataFrame([temp_row])], ignore_index=True)
    df['eNPV'] /= 1000000
    plt.plot(df['p_3eSS'], df['eNPV'])
    plt.show()

    # df.to_csv(path + r'data/{}.csv'.format(csv_name))

    return df
df = gradient_plot_methodol()

def gradient_plot_k2_methodol(csv_name='gradient_plot_k2'):

    df = pd.DataFrame()

    trial = Trial(theta_mu=30, xy_sigma=45,
                  # phase_2_mu=30, phase_2_sigma=15,
                  delta_eff=20, delta_fut=0,
                  market_value=1000000, sensitivity=0.7,
                  general_setup_cost=80000,
                  p2_sample=50, p2_subject_cost=200, p2_setup_cost=30000,
                  p3_subject_cost=800, p3_setup_cost=50000, interim_cost=2000)

    for sample_size in range(400, 816, 16):
        temp_row = trial.utility(k=2, N3=sample_size, n_sims=3000)
        df = pd.concat([df, pd.DataFrame([temp_row])], ignore_index=True)

    plt.plot(df['p_3eSS'], df['eNPV'])
    plt.show()

    df.to_csv(path + r'data/{}.csv'.format(csv_name))

    return df
# df = gradient_plot_k2_methodol()


def gradient_plot_demo_1(csv_name='gradient_plot_demo_1'):
    demo_programmes = pd.read_csv(path + r'data/final_35_abs.csv')

    test_programmes = ['E', 'M', 'P', 'R']
    demo_programmes = demo_programmes[demo_programmes['name'].isin(test_programmes)].reset_index(drop=True)

    plot_df = pd.DataFrame()
    for idx, row in demo_programmes.iterrows():
        temp_programme = Trial(name=row['name'],
                               theta_mu=row['theta_mu'], xy_sigma=row['xy_sigma'],
                               phase_2_mu=None, phase_2_sigma=None,
                               phase_3_mu=None, phase_3_sigma=None,
                               delta_eff=row['delta_eff'], delta_fut=0,
                               market_value=row['market_value'], sensitivity=row['S'],
                               general_setup_cost=row['general_setup_cost'],
                               p2_sample=row['p2_sample'],
                               p2_subject_cost=row['p2_subject_cost'],
                               p2_setup_cost=row['p2_setup_cost'],
                               p3_subject_cost=row['p3_subject_cost'],
                               p3_setup_cost=row['p3_setup_cost'],
                               interim_cost=row['interim_cost'])

        for sample_size in range(400, 1200, 16):
            temp_row = temp_programme.utility(k=2, N3=sample_size, n_sims=3000)
            temp_row.update({'Programme': row['name']})
            plot_df = pd.concat([plot_df, pd.DataFrame([temp_row])], ignore_index=True)

        print(idx)

        plt.plot(plot_df['p_3eSS'], plot_df['eNPV'])
        plt.show()

    plot_df.to_csv(path + r'data/{}.csv'.format(csv_name))

    return plot_df
# plot_df = gradient_plot_demo_1()