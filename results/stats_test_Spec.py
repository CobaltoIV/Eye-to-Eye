import seaborn as sns
from math import sqrt
from scipy.stats import shapiro, mannwhitneyu, levene
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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

    doctor_dic = {'Neurology': ['D1', 'D2', 'D8'],
                  'MedGeral': ['D4', 'D5', 'D7', 'D16'],
                  'Endocrinology': ['D3', 'D10'],
                  'Ginecology': ['D6', 'D9', 'D12', 'D14']}
    doctor_lst = doctor_dic[args.spec]
    df = pd.DataFrame()
    for d in doctor_lst:
        print(d)
        df_presential, df_virtual = get_percentages(d)

        mann_whit_test(df_presential['Patient_Percentage'].round(
            4), df_virtual['Patient_Percentage'].round(4))

        df_d = pd.concat([df_presential[['Patient_Percentage', 'type', 'Doctor']].round(
            4), df_virtual[['Patient_Percentage', 'type', 'Doctor']].round(3)], ignore_index=True)
        df = pd.concat([df, df_d])

    print('Joint Test')
    full_p_data = df.loc[df['type'] == 'Face-to-Face']['Patient_Percentage']
    full_v_data = df.loc[df['type'] == 'Virtual']['Patient_Percentage']

    mann_whit_test(full_p_data, full_v_data)

    ax = sns.violinplot(x='Doctor', y='Patient_Percentage',
                        hue='type', data=df, split=True)

    plt.title(args.spec)
    plt.ylabel("Patient %")
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process gaze output csv file')

    parser.add_argument('-s', '--spec', type=str,
                        help="Medical Specialty")

    args = parser.parse_args()
    print(args)
    main(args)
