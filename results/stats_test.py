import seaborn as sns
from math import sqrt
from scipy.stats import shapiro, mannwhitneyu, levene
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from matplotlib.ticker import PercentFormatter


def normal_test(data):
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
    return


def mann_whit_test(p_data, v_data):
    print("Presential Median: " + str(p_data.median()))
    print("Virtual Median: " + str(v_data.median()))

    U1, p = mannwhitneyu(p_data, v_data,
                         alternative="two-sided")
    print('Statistics=%.3f, p=%.6f' % (U1, p))
    if p < 0.05:
        print('Significant Difference')
        z = (U1 - 200)/(sqrt(400*41/12))
        d = z/sqrt(40)
        print(d)


def get_percentages(d):
    presential_stats = d + '/Presential/proc_res/Totals/Stats.csv'
    virtual_stats = d + '/Virtual/proc_res/Totals/Stats.csv'

    df_presential = pd.read_csv(presential_stats, index_col=0)
    df_virtual = pd.read_csv(virtual_stats, index_col=0)

    df_presential['type'] = 'Face-to-Face'
    df_virtual['type'] = 'Virtual'

    df_presential['Doctor'] = d
    df_virtual['Doctor'] = d

    return df_presential, df_virtual


def main(args):

    df_presential, df_virtual = get_percentages(args.doctor)

    presential_data = df_presential['Patient_Percentage'].round(4)
    virtual_data = df_virtual['Patient_Percentage'].round(4)

    mann_whit_test(presential_data, virtual_data)

    df = pd.concat([df_presential[['Patient_Percentage', 'type']].round(
        4), df_virtual[['Patient_Percentage', 'type']].round(3)], ignore_index=True)
    df['Doctor'] = args.doctor

    ax = sns.violinplot(x='Doctor', y='Patient_Percentage',
                        hue='type', data=df, split=True)

    plt.title(args.doctor)
    plt.ylabel("Patient %")
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process gaze output csv file')

    parser.add_argument('-d', '--doctor', type=str,
                        help="Doctor to perform the test")

    args = parser.parse_args()
    print(args)
    main(args)
