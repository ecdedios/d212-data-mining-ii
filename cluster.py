#!/usr/bin/env python

import sys

# setting the random seed for reproducibility
import random
random.seed(493)

# for manipulating dataframes
import pandas as pd
import numpy as np

# for modeling
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# set display options
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main entry point for the script."""

    # read the csv file
    df = pd.read_csv('churn_clean.csv')
    df.info()
    df.head()

    columns = ['Children',
            'Age',
            'Income',
            'MonthlyCharge',
            'Bandwidth_GB_Year',
            'Tenure']

    # remove outliers in "Children"
    df = df[df['Children'] < 8]
    print(df.shape)

    # drop unnecessary columns not used in the analysis
    df = df[columns]

    # select rows that are duplicated based on all columns
    dup = df[df.duplicated()]

    # find out how many rows are duplicated
    print(dup.shape)

    def show_missing(df):
        """
        Takes a dataframe and returns a dataframe with stats
        on missing and null values with their percentages.
        """
        null_count = df.isnull().sum()
        null_percentage = (null_count / df.shape[0]) * 100
        empty_count = pd.Series(((df == ' ') | (df == '')).sum())
        empty_percentage = (empty_count / df.shape[0]) * 100
        nan_count = pd.Series(((df == 'nan') | (df == 'NaN')).sum())
        nan_percentage = (nan_count / df.shape[0]) * 100
        dfx = pd.DataFrame({'num_missing': null_count, 'missing_percentage': null_percentage,
                            'num_empty': empty_count, 'empty_percentage': empty_percentage,
                            'nan_count': nan_count, 'nan_percentage': nan_percentage})
        return dfx

    print(show_missing(df))

    print(df.head())

    # scale the data
    scaler = StandardScaler()

    # apply scaler() to all the continuous column
    scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    print(scaled.head())

    # save the prepared data set
    scaled.to_csv('churn_prepared1.csv', index=False)

    # determine the best value for k
    inertia = np.array([])
    k_vals = range(1,10)

    for k in k_vals:
        kmeans = KMeans(n_clusters=k, random_state=493)
        kmeans.fit(scaled)
        inertia = np.append(inertia, kmeans.inertia_)

    inertia_vals = pd.DataFrame(inertia, index=k_vals, columns=['Inertia'])

    # do k-means clustering
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=493)
    kmeans.fit(scaled)

    # get predictions
    predictions = kmeans.fit_predict(scaled)

    # calculate the silhouette score
    silhouette = silhouette_score(scaled, predictions)
    print(f'Silhouette Score: {silhouette}, {n_clusters} clusters')

    # assign cluster labels
    df['Cluster'] = kmeans.labels_ + 1

    # calculate cluster summary
    cluster = df.groupby('Cluster').agg(['mean', 'median', 'std']).transpose()
    cluster.columns = ['Cluster 1', 'Cluster 2']
    print(cluster)

    # get the centroids
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_), 
        index=['Cluster 1', 'Cluster 2'], 
        columns=df.columns[:-1]
    )

    # view the centroids
    print('Cluster Centroids:\n')
    for i in centroids.index:
        print(i)
        for col in centroids.columns:
            print(f'{col}: {round(centroids[col][i], 2)}')

if __name__ == '__main__':
    sys.exit(main())










__author__ = "Ednalyn C. De Dios, et al."
__copyright__ = "Copyright 2023, X Project"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ednalyn C. De Dios"
__email__ = "ednalyn.dedios@gmail.com"
__status__ = "Prototype"