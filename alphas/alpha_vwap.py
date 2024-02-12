
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from functions.functions import fill_nan
from functions.functions import *


def process_stock(stock_data):
    if stock_data.empty or stock_data.isnull().values.any():
        # Handle empty or NaN data
        return None
    
    # Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(stock_data)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    # Assign ranks based on cluster membership
    rankings = {i: rank for rank, i in enumerate(clusters)}
    
    # Aggregate rankings and assign final rank
    final_rank = sum(rankings.values()) + 1

    return final_rank



def vwap(book_value):
    total_returns = pd.DataFrame()
    final_rankings_dict = {}
    array = []
    cumulativePnL = 0
    directory = '/Users/thomastan/Documents/Saccade Capital/sample_data'
    for filename in os.listdir(directory):
        if filename.startswith("close_"):
            close = pd.read_csv(directory + '/' + filename)
            close = fill_nan(close)
            suffix = filename.split("_")[-1]
            open = pd.read_csv(directory + '/open_' + suffix)
            open = fill_nan(open)
            high = pd.read_csv(directory + '/high_' + suffix)
            high = fill_nan(high)
            low = pd.read_csv(directory + '/low_' + suffix)
            low = fill_nan(low)
            volume = pd.read_csv(directory + '/volume_' + suffix)
            volume=fill_nan(volume)
            # midp = pd.read_csv(directory + '/midp_' + suffix)
            

            
            for i in range(1, 100):  
                stock_data = pd.concat([close[f"Stock{i}"], open[f"Stock{i}"], high[f"Stock{i}"],
                                        low[f"Stock{i}"], volume[f"Stock{i}"]], axis=1)
                stock_data = stock_data.dropna()
                final_rankings_dict[f"Stock{i}"] = process_stock(stock_data)
                print(final_rankings_dict)

    sorted_stocks = sorted(final_rankings_dict.items(), key=lambda x: x[1])
    return sorted_stocks


#This alpha was a work in progress but was trying to use un-supervised learning here to cluster different stocks together as 'strong long', 'strong short', 'weak long', 'weak short' and 'flat' positions
print(vwap(100000))


    