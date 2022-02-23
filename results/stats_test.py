from ast import arg
import seaborn as sns
from cmath import sqrt
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
   
    df_presential = pd.read_csv(args.presential, index_col=0)
    df_virtual = pd.read_csv(args.virtual, index_col=0)
    
    df_presential['type'] = 'Presential'
    df_virtual['type'] = 'Virtual'
    
    presential_data = df_presential['Patient_Percentage'].round(4)
    virtual_data = df_virtual['Patient_Percentage'].round(4)
    
    
    df =  pd.concat([df_presential[['Patient_Percentage', 'type']].round(4),df_virtual[['Patient_Percentage', 'type']].round(4)], ignore_index=True)
    df['Doctor'] = 'D1'
    normal_test(presential_data)
    normal_test(virtual_data)
    
    print(levene(presential_data,  virtual_data))
    
    print(df)
    U1, p = mannwhitneyu(presential_data, virtual_data, method='auto')
    print('Statistics=%.3f, p=%.3f' % (U1, p))
    U2 = 20*20-U1
    z = (U1 + 0.5) - ((U1+U2)/2)/sqrt((20*20*(20+20+1))/12)
    
    sns.violinplot(y = 'Patient_Percentage', x='Doctor', hue = 'type',data = df, split=True, showmedians=True) 
    
    #plt.boxplot([presential_data, virtual_data])
    #plt.hist(presential_data, bins=30)
    #plt.hist(virtual_data, bins=30)
    
    
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process gaze output csv file')
    
    parser.add_argument('-p', '--presential',type=str, help="Input .csv name")
    parser.add_argument('-v', '--virtual',type=str, help="Input .csv name")
   #parser.add_argument('-n', '--number_clusters', type=int, help="Number of clusters to input")
   
    
    args = parser.parse_args()
    print(args)
    main(args)
    
