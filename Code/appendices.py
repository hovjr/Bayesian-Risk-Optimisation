import numpy as np
import pandas as pd
path = '/Users/alkoulous/Desktop/Bath - MSc/THESIS/Thesis/'

def cat_params():
    display_df = pd.read_csv(path + r'data/programme_data_abs.csv')

    display_df['$G$'] /= 1000000
    display_df['$c_0$'] /= 1000
    display_df['$c_2$'] /= 1000
    display_df['$c_3$'] /= 1000
    display_df['$K$'] /= 1000

    display_df['$c_0$'] = display_df['$c_0$'].astype(int)
    display_df['$c_2$'] = display_df['$c_2$'].astype(int)
    display_df['$c_3$'] = display_df['$c_3$'].astype(int)
    display_df['$N_2$'] = display_df['$N_2$'].astype(int)
    display_df['$N_3$'] = display_df['$N_3$'].astype(int)
    display_df['$K$'] = display_df['$K$'].astype(int)

    display_df.drop(columns=['prior_mu', 'prior_sigma', 'p3_max_sample', '$N_3$', 'eNPV',
                             'assurance', 'upto_p2_cost', 'p3_cost', 'upto_p3_cost', 'total_cost']
                    , inplace=True)

    display_df.to_csv(path + r'data/programme_catalogue.csv', index=False)
    display_df.to_latex(path + r'tables/programme_catalogue.tex',
                              bold_rows=True, float_format="%1.2f", escape=True, index=False)

    return display_df

# catalogue_parameters = cat_params()


def demonstration_1():
    display_df = pd.read_csv(path + r'data/programme_data_abs.csv')
    test_programmes = ['E', 'F', 'H', 'I', 'M', 'O', 'P', 'R']
    display_df = display_df[display_df['Programme'].isin(test_programmes)]

    display_df['total_cost'] /= 1000000

    display_df['eNPV'] /= 1000000
    display_df['$G$'] /= 1000000
    display_df['$c_0$'] /= 1000
    display_df['$c_2$'] /= 1000
    display_df['$c_3$'] /= 1000
    display_df['$K$'] /= 1000

    display_df['$c_0$'] = display_df['$c_0$'].astype(int)
    display_df['$c_2$'] = display_df['$c_2$'].astype(int)
    display_df['$c_3$'] = display_df['$c_3$'].astype(int)
    display_df['$N_2$'] = display_df['$N_2$'].astype(int)
    display_df['$N_3$'] = display_df['$N_3$'].astype(int)
    display_df['$K$'] = display_df['$K$'].astype(int)

    display_df_1 = display_df[['Programme', '$&delta$', '$&theta$', '$&sigma$', '$G$', '$S$',
                               '$c_0$', '$c_2$', '$&gamma_2$', '$N_2$', '$c_3$', '$&gamma_3$',
                               '$N_3$', '$K$']]
    # display_df_1.to_csv(path + r'data/demonstration_1.csv', index=False)
    # display_df_1.to_latex(path + r'tables/demonstration_1.tex',
    #                           bold_rows=True, float_format="%1.2f", escape=True, index=False)

    return display_df_1

demonstration_1 = demonstration_1()

