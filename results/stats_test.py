import seaborn as sns
from math import sqrt
from scipy.stats import shapiro, mannwhitneyu, levene
import matplotlib.pyplot as plt
import argparse
import pandas as pd


def normal_test(data):
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
    return


def main(args):

    presential_stats = args.doctor + '/Presential/proc_res/Totals/Stats.csv'
    virtual_stats = args.doctor + '/Virtual/proc_res/Totals/Stats.csv'

    df_presential = pd.read_csv(presential_stats, index_col=0)
    df_virtual = pd.read_csv(virtual_stats, index_col=0)

    df_presential['type'] = 'Presential'
    df_virtual['type'] = 'Virtual'

    presential_data = df_presential['Patient_Percentage'].round(4)
    virtual_data = df_virtual['Patient_Percentage'].round(4)

    df = pd.concat([df_presential[['Patient_Percentage', 'type']].round(
        4), df_virtual[['Patient_Percentage', 'type']].round(3)], ignore_index=True)
    df['Doctor'] = args.doctor

    print(presential_data.describe())
    print(virtual_data.describe())

    U1, p = mannwhitneyu(presential_data, virtual_data,
                         alternative="two-sided")
    print('Statistics=%.3f, p=%.6f' % (U1, p))
    if p < 0.05:
        print('Significant Difference')
        z = (U1 - 200)/(sqrt(400*41/12))
        d = z/sqrt(40)
        print(d)

    sns.violinplot(y='Patient_Percentage', x='Doctor',
                   hue='type', data=df, split=True, showmedians=True)

    plt.title(args.doctor)
    plt.ylabel("Patient %")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process gaze output csv file')

    #parser.add_argument('-p', '--presential', type=str, help="Input .csv name")
    #parser.add_argument('-v', '--virtual', type=str, help="Input .csv name")
    parser.add_argument('-d', '--doctor', type=str,
                        help="Doctor to perform the test")

    args = parser.parse_args()
    print(args)
    main(args)
