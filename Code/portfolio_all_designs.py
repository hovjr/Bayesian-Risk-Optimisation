import time
import pandas as pd
path = '/Users/alkoulous/Desktop/Bath - MSc/THESIS/Thesis/'

def kp3(dev_costs, eNPV, budget, n, selected_names):
    if n == 0 or budget == 0:
        return 0, []

    selected_names_tuple = tuple(selected_names)

    if (n, budget, selected_names_tuple) in cached_s:
        return cached_s[(n, budget, selected_names_tuple)]

    if names[n - 1] in selected_names:
        result = kp3(dev_costs, eNPV, budget, n - 1, selected_names)
    else:
        if dev_costs[n - 1] <= budget:
            eNPV_with_current, selected_with_current = kp3(dev_costs, eNPV,
                                                           budget - dev_costs[n - 1], n - 1,
                                                           selected_names + [names[n - 1]])
            eNPV_with_current += eNPV[n - 1]

            eNPV_without_current, selected_without_current = kp3(dev_costs, eNPV, budget, n - 1,
                                                                 selected_names)

            if eNPV_with_current > eNPV_without_current:
                result = [eNPV_with_current, selected_with_current + [n - 1]]
            else:
                result = [eNPV_without_current, selected_without_current]
        else:
            result = kp3(dev_costs, eNPV, budget, n - 1, selected_names)

    cached_s[(n, budget, selected_names_tuple)] = result

    return result


def ranked_ass():
    file = 'single_design_demo_1'
    all_designs = pd.read_csv(path + r'data/{}.csv'.format(file))
    all_designs.sort_values(by='eNPV', ascending=False, inplace=True)
    all_designs['cum_cost'] = all_designs['total_cost'].cumsum()
    all_designs['cum_eNPV'] = all_designs['eNPV'].cumsum()
    return all_designs[all_designs['cum_cost']<2000000]
# ranked_ass()

########## CASES ##########

file = 'demo_3_p3designs'
# demo_3_df, demo_3_p3designs
all_designs = pd.read_csv(path + r'data/{}.csv'.format(file))
# all_designs = all_designs.sort_values(by=['eNPV'], ascending=False)
# all_designs = pd.merge(all_designs, all_designs.drop_duplicates(subset=['Programme'], keep='first')[['Programme', 'N_3_max']], on='Programme', how='left')
# all_designs = all_designs[all_designs['N_3_max_x']<=all_designs['N_3_max_y']]
# all_designs = all_designs.rename(columns={'N_3_max_x':'N_3_max'}).drop(columns=['N_3_max_y'])
# all_designs.sort_values(by='Programme', inplace=True)
# all_designs.to_csv(path + r'data/{}.csv'.format(file), index=False)
# all_designs = all_designs.drop_duplicates(subset=['Programme'], keep='first')

all_designs.sort_values(by=['Programme'], inplace=True)
names = all_designs['Programme'].to_list()

# dev_costs = all_designs['upto_p2_cost_neutral'].to_list()
# eNPV = all_designs['eNPV_neutral'].to_list()

# dev_costs = all_designs['upto_p2_cost'].to_list()
eNPV = all_designs['eNPV'].to_list()
dev_costs = all_designs['p3_cost'].to_list()

# all_designs['N_3_max'] = all_designs['Programme']
n_3 = all_designs['N_3_max'].to_list()

# check whether mil or dec
# B = 1000000
B = 1500000 - 995420
n = len(eNPV)

start = time.process_time()

cached_s = {}
optimal_eNPV, optimal_selection = kp3(dev_costs, eNPV, B, n, [])
selected_combinations = [(names[i], n_3[i]) for i in optimal_selection]

print("optimal enpv", optimal_eNPV)
print("selected programmes", [names[i] for i in optimal_selection])
print("selected designs:", [(names[i], n_3[i]) for i in optimal_selection])
print("total cost ", sum([dev_costs[i] for i in optimal_selection]))
print("time ", time.process_time() - start)
cached_s = {}

# opt_design = pd.read_csv(path + r'data/demo_3_p3designs.csv'.format(file))
# opt_design = opt_design.drop_duplicates(subset=['Programme'], keep='first')
# opt_design = opt_design[['Programme', 'N_3_max', 'p3_cost', 'eNPV']]
# final_df = pd.DataFrame()
# for items in selected_combinations:
#     temp_row = all_designs[(all_designs['Programme'] == items[0]) & (all_designs['N_3_max'] == items[1])]
#     final_df = pd.concat([final_df, temp_row], ignore_index=True)
# final_df = pd.merge(final_df, opt_design, on='Programme')
# final_df = final_df[['Programme', 'N_3_max_y', 'p3_cost_y', 'eNPV_y', 'p_3eSS', 'p3_cost_x', 'eNPV_x']]
# final_df = final_df.sort_values(by='Programme')
# final_df['eNPV_y'] /= 1000000
# final_df['p3_cost_y'] /= 1000000
# final_df['eNPV_x'] /= 1000000
# final_df['p3_cost_x'] /= 1000000
# final_df.to_latex(path + r'tables/demo_3_d2_1.tex'.format(file),
#                           bold_rows=True, float_format="%1.3f", escape=True, index=False)
