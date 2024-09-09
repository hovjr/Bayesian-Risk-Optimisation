import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
path = '/Users/alkoulous/Desktop/Bath - MSc/THESIS/Thesis/'

def power_vs_n(png_name='power_vs_n'):
    data = pd.read_csv(path + r'data/{}.csv'.format(png_name))
    data = data.rename(columns={"Sample size": "N", "Power": "power"})

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    sns.lineplot(x="N", y="power", color='black', data=data, errorbar=None)

    plt.plot([0, 214], [0.9, 0.9], color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.plot([214, 214], [0, 0.9], color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

    plt.ylim(0, 1.05)
    plt.xlim(0, 400)

    x_ticks = list(range(0, 600, 100))
    plt.xticks(ticks=x_ticks, labels=[str(x) if x != 200 else "" for x in x_ticks], fontsize=14)
    plt.text(214, -0.05, '214', va='center', ha='center', fontsize=14, color='black')

    plt.yticks(ticks=[x / 100 for x in range(25, 125, 25)], fontsize=14)
    plt.text(-7, 0.9, '0.90', va='center', ha='right', fontsize=14, color='black')

    plt.xlabel("N", fontsize=17)
    plt.ylabel("Power (1-β)", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(path + r'figs/{}.png'.format(png_name))
    plt.show()
# power_vs_n()

def zeta_plot(png_name='zeta_plot'):
    data = pd.read_csv(path + r'data/zeta_plot.csv')
    data = data.rename(columns={"eNPV": "eNPV", "p_3eSS": "p3ess", "Sensitivity": "S"})
    data['S'] = pd.to_numeric(data['S'])
    blue_cmap = LinearSegmentedColormap.from_list("custom_blues", ["#4a90e2", "#002f6c"])
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    sns.lineplot(x="theta", y="zeta", hue="S", palette=blue_cmap, data=data, errorbar=None)
    plt.legend(title="S", bbox_to_anchor=(1.05, 1), loc='upper left',
               title_fontsize=17, fontsize=14, borderaxespad=0.)

    plt.ylim(-0.05, 2.05)
    plt.xlim(10, 30)
    plt.xticks(ticks=range(10, 35, 5), fontsize=14)
    plt.yticks(ticks=[x / 10 for x in range(0, 25, 5)], fontsize=14)

    plt.xlabel("θ", fontsize=17)
    plt.ylabel("ζ(θ)", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(path + r'figs/{}.png'.format(png_name))
    plt.show()
# zeta_plot()

def vary_s(png_name='vary_s'):
    data = pd.read_csv(path + r'data/eNPV_vary_S.csv', index_col=0)
    data = data.rename(columns={"eNPV": "eNPV", "p_3eSS": "p3ess", "S": "S"})
    data['S'] = pd.to_numeric(data['S'])
    blue_cmap = LinearSegmentedColormap.from_list("custom_blues", ["#4a90e2", "#002f6c"])
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    sns.lineplot(x="p3ess", y="eNPV", hue="S", palette=blue_cmap, data=data, errorbar=None)
    data['eNPV'] /= 1000000
    plt.legend(title="S", bbox_to_anchor=(1.05, 1), loc='upper left',
               title_fontsize=17, fontsize=14, borderaxespad=0.)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlim(0, 300)
    plt.xlabel("N3", fontsize=17)
    plt.ylabel("eNPV (£m)", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(path + r'figs/{}.png'.format(png_name))

    plt.show()
# vary_s()


def gradient_plot_single(png_name='gradient_plot_1'):
    data = pd.read_csv(path + r'data/{}.csv'.format(png_name))
    data = data.rename(columns={"p_3eSS": "N", "eNPV": "eNPV"})
    data['eNPV'] /= 1000000

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    sns.lineplot(x="N", y="eNPV", color='black', data=data, errorbar=None)

    plt.axhline(y=0.89, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

    plt.ylim(0, 0.9)
    plt.xlim(0, 350)
    plt.xticks(ticks=range(0, 400, 100), fontsize=14)
    plt.yticks(ticks=[x / 100 for x in range(25, 125, 25)] + [0.89], fontsize=14)

    plt.xlabel("N3", fontsize=17)
    plt.ylabel("eNPV (£m)", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(path + r'figs/{}.png'.format(png_name))
    plt.show()
# gradient_plot_single()

def gradient_plot_single_k2(png_name='gradient_plot_k2'):
    data = pd.read_csv(path + r'data/{}.csv'.format(png_name))
    data = data.rename(columns={"p_3eSS": "N", "eNPV": "eNPV"})
    data['eNPV'] /= 1000000
    data['n3_max'] = data['N'] * 2

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    sns.lineplot(x="n3_max", y="eNPV", color='black', data=data, errorbar=None)

    plt.axvline(x=656, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

    # plt.ylim(1.4, 1.562)
    plt.xlim(400, 800)
    plt.xticks(ticks=list(range(400, 900, 100)) + [656], fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("Maximum N3", fontsize=17)
    plt.ylabel("eNPV (£m)", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(path + r'figs/{}.png'.format(png_name))
    plt.show()
# gradient_plot_single_k2()

def gradient_plot_demo1(png_name='gradient_plot_demo_1'):
    data = pd.read_csv(path + r'data/{}.csv'.format(png_name))
    data = data.rename(columns={"eNPV": "eNPV", "p_3eSS": "p3ess", "S": "S"})
    data['n3_max'] = data['p3ess'] * 2

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    sns.lineplot(x="p3ess", y="eNPV", hue='Programme', data=data, errorbar=None, palette="Set2")
    data['eNPV'] /= 1000000
    plt.legend(title="Programme", bbox_to_anchor=(1.05, 1), loc='upper left',
               title_fontsize=17, fontsize=14, borderaxespad=0.)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlim(200, 600)
    plt.xlabel("Maximum N3", fontsize=17)
    plt.ylabel("eNPV (£m)", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 1])

    # plt.savefig(path + r'figs/{}.png'.format(png_name))

    plt.show()
gradient_plot_demo1()