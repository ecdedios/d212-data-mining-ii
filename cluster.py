#!/usr/bin/env python

import sys

import warnings
warnings.filterwarnings('ignore')

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

def main():
    """Main entry point for the script."""

    # read the csv file
    df = pd.read_csv('churn_clean.csv')

    columns = ['Children',
            'Age',
            'Income',
            'MonthlyCharge',
            'Bandwidth_GB_Year',
            'Tenure']

    # remove outliers in "Children"
    df = df[df['Children'] < 8]

    # drop unnecessary columns not used in the analysis
    df = df[columns]

    # scale the data
    scaler = StandardScaler()

    # apply scaler() to all the continuous column
    scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # save the prepared data set
    scaled.to_csv('churn_prepared1.csv', index=False)

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
        print()

if __name__ == '__main__':
    sys.exit(main())










__author__ = "Ednalyn C. De Dios, et al."
__copyright__ = "Copyright 2023, X Project"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ednalyn C. De Dios"
__email__ = "ednalyn.dedios@gmail.com"
__status__ = "Prototype"