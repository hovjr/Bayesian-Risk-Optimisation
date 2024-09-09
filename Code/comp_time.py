path = '/Users/alkoulous/Desktop/Bath - MSc/THESIS/Thesis/'

########## CASES ##########
all_programmes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
       'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI']

proglist = []
comp_time = pd.DataFrame()
for programme in all_programmes:

    proglist.append(programme)

    all_designs = pd.read_csv(path + r'data/all_designs_35.csv')

    all_designs = all_designs[all_designs['Programme'].isin(proglist)]

    names = all_designs['Programme'].to_list()
    eNPV = all_designs['eNPV'].to_list()
    dev_costs = all_designs['total_cost'].to_list()
    n_3 = all_designs['N_3_max'].to_list()

    B = 2000000
    n = len(eNPV)

    start = time.process_time()

    cached_s = {}
    optimal_eNPV, optimal_selection = kp3(dev_costs, eNPV, B, n, [])
    selected_combinations = [(names[i], n_3[i]) for i in optimal_selection]

    print("Optimal eNPV:", optimal_eNPV)
    print("Selected combinations (name, n3):", selected_combinations)

    total_time = time.process_time()-start
    # print(total_time)

    temp_row = {'Number of programmes': len(proglist), 'Total designs': len(all_designs),
                'Total time': total_time}

    comp_time = pd.concat([comp_time, pd.DataFrame([temp_row])], ignore_index=True)

    comp_time.to_csv(path + r'data/comp_time.csv', index=False)
    comp_time.to_latex(path + r'tables/comp_time.tex',
                              bold_rows=True, float_format="%1.2f", escape=True, index=False)

    # final_df = pd.DataFrame()
    # for items in selected_combinations:
    #     temp_row = all_designs[(all_designs['Programme'] == items[0]) & (all_designs['N_3_max'] == items[1])]
    #     final_df = pd.concat([final_df, temp_row], ignore_index=True)
