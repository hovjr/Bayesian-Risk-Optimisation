import time
import pandas as pd
path = '/Users/alkoulous/Desktop/Bath - MSc/THESIS/Thesis/'

def kp3(dev_costs_p2, dev_costs_p3, eNPV, budget, n, selected_names):
    if n == 0 or budget == 0:
        return 0, []

    selected_names_tuple = tuple(selected_names)

    if (n, budget, selected_names_tuple) in cached_s:
        return cached_s[(n, budget, selected_names_tuple)]

    if names[n - 1] in selected_names:
        result = kp3(dev_costs_p2, dev_costs_p3, eNPV, budget, n - 1, selected_names)
    else:
        if dev_costs_p2[n - 1] + dev_costs_p3[n - 1] <= budget:
            eNPV_with_current, selected_with_current = kp3(dev_costs_p2, dev_costs_p3, eNPV,
                                                           budget - (dev_costs_p2[n - 1] + dev_costs_p3[n - 1]), n - 1,
                                                           selected_names + [names[n - 1]])
            eNPV_with_current += eNPV[n - 1]

            eNPV_without_current, selected_without_current = kp3(dev_costs_p2, dev_costs_p3, eNPV, budget, n - 1,
                                                                 selected_names)

            if eNPV_with_current > eNPV_without_current:
                result = [eNPV_with_current, selected_with_current + [n - 1]]
            else:
                result = [eNPV_without_current, selected_without_current]
        else:
            result = kp3(dev_costs_p2, dev_costs_p3, eNPV, budget, n - 1, selected_names)

    cached_s[(n, budget, selected_names_tuple)] = result

    return result

file = 'demo_3_df'
all_designs = pd.read_csv(rpath + r'data/{}.csv'.format(file))

names = all_designs['Programme'].to_list()
eNPV = all_designs['eNPV_neutral'].to_list()
dev_costs_p2 = all_designs['upto_p2_cost_neutral'].to_list()
dev_costs_p3 = all_designs['p3_cost_neutral'].to_list()

B = 2000000
n = len(eNPV)

start = time.process_time()

cached_s = {}
optimal_eNPV, optimal_selection = kp3(dev_costs_p2, dev_costs_p3, eNPV, B, n, [])

optimal_programmes = [names[i] for i in optimal_selection]

print("total eNPV:", optimal_eNPV)
print("total budget", sum([dev_costs_p2[i] + dev_costs_p3[i] for i in optimal_selection]))
print("programmes", optimal_programmes)
print("time:", time.process_time() - start)

final_df = all_designs[all_designs['Programme'].isin(optimal_programmes)]

# final_df = final_df[['Programme', 'p_3eSS', 'total_cost', 'eNPV']]
# final_df['total_cost'] /= 1000000
# final_df['eNPV'] /= 1000000
# final_df.to_latex(rpath + r'tables/{}.tex'.format(file),
#                           bold_rows=True, float_format="%1.2f", escape=True, index=False)