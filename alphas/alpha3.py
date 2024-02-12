
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from functions.functions import fill_nan
from functions.functions import *
import math



def alpha(book_value):
    total_returns = pd.DataFrame()
    final_rankings_dict = {}
    array = []
    cumulativePnL = 0
    directory = '/Users/thomastan/Documents/Saccade Capital/sample_data'
    for filename in os.listdir(directory):
        if filename.startswith("close_"):
            close = pd.read_csv(directory + '/' + filename)
            minutes = close.Minutes
            close = fill_nan(close)
            close = close.drop(columns=['Minutes'])
            suffix = filename.split("_")[-1]
            open = pd.read_csv(directory + '/open_' + suffix)
            open = fill_nan(open)
            open = open.drop(columns=['Minutes'])
            high = pd.read_csv(directory + '/high_' + suffix)
            high = fill_nan(high)
            high = high.drop(columns=['Minutes'])
            low = pd.read_csv(directory + '/low_' + suffix)
            low = fill_nan(low)
            low = low.drop(columns=['Minutes'])
            volume = pd.read_csv(directory + '/volume_' + suffix)
            volume = fill_nan(volume)
            volume = volume.drop(columns=['Minutes'])
            

            position = ((high + low) / 2 - close) / volume
            ranking = (position.rank(axis=1, ascending=True, method='first') - 50) / 99
            ranking = pd.concat([minutes, ranking], axis=1)
            alpha = normalize_df(ranking, scale=book_value)


            close = pd.read_csv(directory + '/' + filename)
            returns = calculate_returns(alpha, close)
            array.append(float(sum(returns)) - book_value)
            book_value = float(sum(returns)) - book_value
            cumulativePnL += sum(returns)

            total_returns = pd.concat([total_returns, returns], ignore_index=True)
            

    print("Cumulative", cumulativePnL)
    return (total_returns,array)

print(alpha(100000))
print(stats(alpha(100000), 100000))



    