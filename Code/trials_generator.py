from THESIS.Code.Trial import Trial
import pandas as pd
import math
path = '/Users/alkoulous/Desktop/Bath - MSc/THESIS/Thesis/'

def process_raw():
    raw_programmes = pd.read_csv(path + r'data/35_raw.csv')
    for idx, row in raw_programmes.iterrows():
        temp_programme = Trial(name=row['Programme'],
                               theta_mu=row['$&theta$'], xy_sigma=row['$&sigma$'],
                               delta_eff=row['$&delta$'], delta_fut=0,
                               market_value=row['$G$'], sensitivity=row['$S$'],
                               general_setup_cost=row['$c_0$'], p2_sample=None,
                               p2_subject_cost=row['$&gamma_2$'], p2_setup_cost=row['$c_2$'],
                               p3_subject_cost=row['$&gamma_3$'], p3_setup_cost=row['$c_3$'],
                               interim_cost=row['$K$'])

        raw_programmes.loc[idx, 'prior_mu'] = temp_programme.theta_mu
        raw_programmes.loc[idx, 'prior_sigma'] = temp_programme.xy_sigma

        p2_sample = temp_programme.suggest_N2()[0]
        raw_programmes.loc[idx, '$N_2$'] = p2_sample

        custom_opt = temp_programme.custom_opt_p3(k=2, n_sims=3000)

        raw_programmes.loc[idx, '$N_3$'] = custom_opt['p_3eSS']
        raw_programmes.loc[idx, 'p3_max_sample'] = custom_opt['P3 Sample']
        raw_programmes.loc[idx, 'assurance'] = custom_opt['assurance']
        raw_programmes.loc[idx, 'p3_cost'] = custom_opt['p3_cost']
        raw_programmes.loc[idx, 'upto_p2_cost'] = custom_opt['upto_p2_cost']
        raw_programmes.loc[idx, 'total_cost'] = custom_opt['total_cost']

        raw_programmes.loc[idx, 'eNPV'] = custom_opt['eNPV']

        # print(raw_programmes)
        print(idx)

    raw_programmes['$K$'] *= 10
    raw_programmes['$&gamma_2$'] *= 3
    raw_programmes['$&gamma_3$'] *= 0.8
    raw_programmes['$&gamma_3$'] = raw_programmes['$&gamma_3$'].astype(int)

    raw_programmes.to_csv(path + r'data/programme_data_abs.csv', index=False)

    final_35_abs = raw_programmes.copy()
    final_35_abs.columns = ['name', 'theta_mu', 'xy_sigma', 'delta_eff',
                            'prior_mu', 'prior_sigma', 'market_value', 'S',
                            'general_setup_cost', 'p2_setup_cost', 'p2_subject_cost',
                            'p2_sample', 'p3_setup_cost', 'p3_subject_cost',
                            'p3_max_sample', 'p3_ess', 'interim_cost', 'eNPV',
                            'assurance', 'upto_p2_cost', 'p3_cost', 'total_cost']

    final_35_abs.to_csv(path + r'data/final_35_abs.csv', index=False)

    return final_35_abs

# df = process_raw()


def demo_1(demo_programmes, demo, mu_sigma, k_int):

    z_ij = pd.DataFrame()
    for idx, row in demo_programmes.iterrows():
        temp_programme = Trial(name=row['name'],
                               theta_mu=row['theta_mu'],
                               xy_sigma=row['xy_sigma'],
                               delta_eff=row['delta_eff'], delta_fut=0,
                               market_value=row['market_value'],
                               general_setup_cost=row['general_setup_cost'],
                               p2_sample=row['p2_sample'],
                               p2_subject_cost=row['p2_subject_cost'],
                               p2_setup_cost=row['p2_setup_cost'],
                               p3_subject_cost=row['p3_subject_cost'],
                               p3_setup_cost=row['p3_setup_cost'],
                               interim_cost=row['interim_cost'])

        temp_programme.phase_3_mu = temp_programme.phase_3_mu * mu_sigma[0]
        temp_programme.phase_3_sigma = temp_programme.phase_3_sigma * mu_sigma[1]

        custom_opt = temp_programme.custom_opt_p3(k=k_int)
        for n3_max in range(custom_opt['P3 Sample'], math.floor(custom_opt['P3 Sample'] * 0.2), -2):
            temp_util = temp_programme.utility(k=k_int, N3=n3_max, n_sims=1000)
            temp_row = {'Programme': row['name'],
                        'p2_mu': temp_programme.theta_mu, 'p2_sigma': temp_programme.xy_sigma,
                        'p3_mu': temp_programme.phase_3_mu, 'p3_sigma': temp_programme.phase_3_sigma,
                        'N_3_max': n3_max}
            temp_row.update(temp_util)
            z_ij = pd.concat([z_ij, pd.DataFrame([temp_row])], ignore_index=True)
            z_ij = z_ij.sort_values(by=['eNPV'], ascending=False)

        print(idx)

    z_ij.to_csv(path + r'data/all_designs_{}.csv'.format(demo), index=False)
    single_design = z_ij.drop_duplicates(subset=['Programme'], keep='first')
    single_design.to_csv(path + r'data/single_design_{}.csv'.format(demo), index=False)

    return z_ij, single_design

def demo_2_df(k_int=1, demo='demo3'):
    demo_programmes = pd.read_csv(path + r'data/final_35_abs.csv')
    # demo_programmes = demo_programmes.sample(n=8).sort_index()
    test_programmes = ['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI']
    demo_programmes = demo_programmes[demo_programmes['name'].isin(test_programmes)]
    demo_1(demo_programmes, demo='{}_neutral'.format(demo), mu_sigma=[1, 1], k_int=k_int)
    demo_1(demo_programmes, demo='{}_good'.format(demo), mu_sigma=[1.2, 0.8], k_int=k_int)
    demo_1(demo_programmes, demo='{}_bad'.format(demo), mu_sigma=[0.85, 1.1], k_int=k_int)

    neutral_df = pd.read_csv(path + r'data/single_design_{}_neutral.csv'.format(demo),
                             usecols=['Programme', 'upto_p2_cost', 'p3_mu', 'p3_sigma', 'p_3eSS', 'p3_cost', 'eNPV'])
    neutral_df = neutral_df.add_suffix('_neutral').rename(columns={'Programme_neutral': 'Programme'})

    good_df = pd.read_csv(path + r'data/single_design_{}_good.csv'.format(demo),
                          usecols=['Programme', 'p3_mu', 'p3_sigma', 'p_3eSS', 'p3_cost', 'eNPV'])
    good_df = good_df.add_suffix('_good').rename(columns={'Programme_good': 'Programme'})

    bad_df = pd.read_csv(path + r'data/single_design_{}_bad.csv'.format(demo),
                          usecols=['Programme', 'p3_mu', 'p3_sigma', 'p_3eSS', 'p3_cost', 'eNPV'])
    bad_df = bad_df.add_suffix('_bad').rename(columns={'Programme_bad': 'Programme'})

    demo_df = pd.merge(neutral_df, good_df, how='inner', on='Programme')
    demo_df = pd.merge(demo_df, bad_df, how='inner', on='Programme')

    demo_df = demo_df[['Programme',
                           'p3_mu_neutral', 'p3_sigma_neutral', 'upto_p2_cost_neutral', 'p3_cost_neutral', 'eNPV_neutral',
                           'p3_mu_good', 'p3_sigma_good', 'p3_cost_good', 'eNPV_good',
                           'p3_mu_bad', 'p3_sigma_bad', 'p3_cost_bad', 'eNPV_bad']]
    demo_df.to_csv(path + r'data/{}_df.csv'.format(demo), index=False)

    display_demo = demo_df.copy()
    display_demo['upto_p2_cost_neutral'] /= 1000000
    display_demo['upto_p2_cost_neutral'] = display_demo['upto_p2_cost_neutral']
    display_demo['p3_cost_neutral'] /= 1000000
    display_demo['p3_cost_neutral'] = display_demo['p3_cost_neutral']
    display_demo['eNPV_neutral'] /= 1000000
    display_demo['prior_neutral'] = (
                '$N(' + display_demo['p3_mu_neutral'].map('{:.0f}'.format) +
                ', ' + display_demo['p3_sigma_neutral'].map('{:.0f}'.format) + '&2)$')

    display_demo['p3_cost_good'] /= 1000000
    display_demo['p3_cost_good'] = display_demo['p3_cost_good']
    display_demo['eNPV_good'] /= 1000000
    display_demo['prior_good'] = ('$N(' + display_demo['p3_mu_good'].map('{:.0f}'.format) +
                                    ', ' + display_demo['p3_sigma_good'].map(
                '{:.0f}'.format) + '&2)$')

    display_demo['p3_cost_bad'] /= 1000000
    display_demo['p3_cost_bad'] = display_demo['p3_cost_bad']
    display_demo['eNPV_bad'] /= 1000000
    display_demo['prior_bad'] = ('$N(' + display_demo['p3_mu_bad'].map('{:.0f}'.format) +
                                   ', ' + display_demo['p3_sigma_bad'].map(
                '{:.0f}'.format) + '&2)$')

    display_demo = display_demo[['Programme',
                           'prior_neutral', 'upto_p2_cost_neutral', 'p3_cost_neutral', 'eNPV_neutral',
                           'prior_good', 'p3_cost_good', 'eNPV_good',
                           'prior_bad', 'p3_cost_bad', 'eNPV_bad']]

    display_demo.to_latex(path + r'tables/display_{}.tex'.format(demo),
                              bold_rows=True, float_format="%1.3f", escape=True, index=False)

    return demo_df
demo_3_df = demo_2_df(k_int=1, demo='demo_3')

def display_demo_3():
    display_demo_3 = pd.read_csv(path + r'data/demo_3_df.csv')
    # opt_programmes = ['AA', 'AB', 'AC', 'AF', 'AG', 'AH', 'AI']
    # demo_programmes = display_demo_3[display_demo_3['Programme'].isin(opt_programmes)]
    display_demo_3['upto_p2_cost_neutral'] /= 1000000
    display_demo_3['upto_p2_cost_neutral'] = display_demo_3['upto_p2_cost_neutral']
    display_demo_3['p3_cost_neutral'] /= 1000000
    display_demo_3['p3_cost_neutral'] = display_demo_3['p3_cost_neutral']
    display_demo_3['eNPV_neutral'] /= 1000000
    display_demo_3['prior_neutral'] = (
                '$N(' + display_demo_3['p3_mu_neutral'].map('{:.0f}'.format) +
                ', ' + display_demo_3['p3_sigma_neutral'].map('{:.0f}'.format) + '&2)$')


    display_demo_3['p3_cost_good'] /= 1000000
    display_demo_3['p3_cost_good'] = display_demo_3['p3_cost_good']
    display_demo_3['eNPV_good'] /= 1000000
    display_demo_3['prior_good'] = ('$N(' + display_demo_3['p3_mu_good'].map('{:.0f}'.format) +
                                    ', ' + display_demo_3['p3_sigma_good'].map(
                '{:.0f}'.format) + '&2)$')

    display_demo_3['p3_cost_bad'] /= 1000000
    display_demo_3['p3_cost_bad'] = display_demo_3['p3_cost_bad']
    display_demo_3['eNPV_bad'] /= 1000000
    display_demo_3['prior_bad'] = ('$N(' + display_demo_3['p3_mu_bad'].map('{:.0f}'.format) +
                                   ', ' + display_demo_3['p3_sigma_bad'].map(
                '{:.0f}'.format) + '&2)$')

    for idx, row in display_demo_3.iterrows():
        if row['Programme'] in ['AH', 'AB', 'AC', 'AF']:
            display_demo_3.loc[idx, 'd2_prior'] = row['prior_good']
            display_demo_3.loc[idx, 'd2_c3'] = row['p3_cost_good']
            display_demo_3.loc[idx, 'd2_eNPV'] = row['eNPV_good']

        elif row['Programme'] in ['AI', 'AA', 'AE']:
            display_demo_3.loc[idx, 'd2_prior'] = row['prior_bad']
            display_demo_3.loc[idx, 'd2_c3'] = row['p3_cost_bad']
            display_demo_3.loc[idx, 'd2_eNPV'] = row['eNPV_bad']

        # elif row['Programme'] in ['AD', 'AG']:
        #     display_demo_3.loc[idx, 'd2_prior'] = row['prior_bad']
        #     display_demo_3.loc[idx, 'd2_c3'] = row['p3_cost_bad']
        #     display_demo_3.loc[idx, 'd2_eNPV'] = row['eNPV_bad']

    all_good = pd.read_csv(path + r'data/all_designs_demo_3_good.csv')
    all_good = all_good[all_good['Programme'].isin(['AH', 'AB', 'AC', 'AF'])]

    all_bad = pd.read_csv(path + r'data/all_designs_demo_3_bad.csv')
    all_bad = all_bad[all_bad['Programme'].isin(['AI', 'AA', 'AE'])]

    all_neutral = pd.read_csv(path + r'data/all_designs_demo_3_neutral.csv')
    all_neutral = all_neutral[all_neutral['Programme'].isin(['AD', 'AG'])]

    all_designs_demo_3 = pd.concat([all_good, all_bad], ignore_index=True)
    all_designs_demo_3 = pd.concat([all_designs_demo_3, all_neutral], ignore_index=True)
    all_designs_demo_3.to_csv(path + r'data/demo_3_p3designs.csv', index=False)

    display_demo_3 = display_demo_3[['Programme', 'prior_neutral', 'upto_p2_cost_neutral',
                                     'p3_cost_neutral', 'eNPV_neutral', 'd2_prior', 'd2_c3',
                                     'd2_eNPV']]

    display_demo_3 = display_demo_3.sort_values(by='Programme')

    display_demo_3.to_latex(path + r'tables/display_{}.tex'.format('demo_3'),
                              bold_rows=True, float_format="%1.3f", escape=True, index=False)

    display_demo_3['diff_enpv'] = display_demo_3['eNPV_neutral'] / display_demo_3['d2_eNPV']
    display_demo_3['diff_cost'] = display_demo_3['p3_cost_neutral'] / display_demo_3['d2_c3']

    return display_demo_3
# display_demo_3 = display_demo_3()
# display_demo_3['Programme'].unique()
