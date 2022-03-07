from sklearn.cluster import KMeans
import matplotlib.pyplot as plot
import seaborn as sns
import argparse
import pandas as pd

def main(args):
    df = pd.read_csv(args.input_file,index_col=0)
    gaze  = df [['gaze360_x', 'gaze360_y']]
    gaze = gaze.dropna(axis=0)
    
    gaze = gaze.drop(gaze[(abs(gaze['gaze360_x']) > 5) | (abs(gaze['gaze360_y']) > 5)].index)
    """
    kmeans = KMeans(n_clusters = args.number_clusters, random_state=0).fit(gaze[['gaze360_x', 'gaze360_y']])
    print(kmeans.cluster_centers_)
    plot.scatter(gaze['gaze360_x'], gaze['gaze360_y'], c=kmeans.labels_)
    plot.scatter(kmeans.cluster_centers_[1,:],kmeans.cluster_centers_[0,:], marker = '^', c = '#FF0000')
    plot.gca().invert_yaxis()
    plot.show()
    """
    sns.histplot(gaze, x= 'gaze360_x', y='gaze360_y')
    plot.gca().invert_yaxis()
    plot.show()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process gaze output csv file')
    
    parser.add_argument('-i', '--input_file',type=str, help="Input .csv name")
    parser.add_argument('-n', '--number_clusters', type=int, help="Number of clusters to input")
   
    
    args = parser.parse_args()
    print(args)
    main(args)
    
