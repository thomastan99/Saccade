
import pandas as pd
import numpy as np
import os

from functions.functions import *

def alpha_baseline_reversion(book_value):
    total_returns = pd.DataFrame()
    array = []
    cumulativePnL = 0
    directory = '/Users/thomastan/Documents/Saccade Capital/sample_data'
    for filename in os.listdir(directory):
        if filename.startswith("close_"):
            close = pd.read_csv(directory+'/'+filename)
            ranked_data = ranking(close, reversion=1)
            
            alpha = normalize_df(ranked_data, scale=book_value)
            returns = calculate_returns(alpha, close)
            array.append(float(sum(returns)) - book_value)
            book_value = float(sum(returns)) - book_value
            cumulativePnL += sum(returns)
            total_returns = pd.concat([total_returns, returns], ignore_index=True)


    print("Cumulative", cumulativePnL)
    return (total_returns,array)