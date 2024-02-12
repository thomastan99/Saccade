import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
def fill_nan(df):
    cleaned_df = df.fillna(method='ffill')
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

def ranking(df, window=5, universe_size=100, reversion = 1):
    #Logic of forward filling might need to be re-thought out
    minutes = df.Minutes
    df = df.drop(columns=['Minutes'])
    returns_df = df.pct_change(fill_method='ffill')
    rolling_returns = returns_df.rolling(window).mean()
    valid_returns = rolling_returns[~rolling_returns.isna() & ~np.isinf(rolling_returns)]
    reversion_signals = -reversion *(valid_returns > 0).astype(int)

    reversion_rankings = (reversion_signals.rank(axis=1, ascending=True, method='first') - 50)/universe_size
    return pd.concat([minutes, reversion_rankings], axis=1)

def ranking_transformation(df, power=3):
    minutes = df.Minutes
    df = df.drop(columns=['Minutes'])
    df = df.applymap(lambda x: x**power if power % 2 != 0 else x**power if x >= 0 else -(x**power))
    return pd.concat([minutes, df], axis=1)



def normalize_row(row, scale):
    neg_values = row[row < 0]
    pos_values = row[row >= 0]
    pos_sum = 0
    neg_sum=0
    if len(neg_values) > 0:
        neg_sum = neg_values.sum()
        neg_norm = -neg_values / neg_sum
    else:
        neg_norm = pd.Series([0] * len(row))

    if len(pos_values) > 0:
        pos_sum = pos_values.sum()
        pos_norm = pos_values / pos_sum
    else:
        pos_norm = pd.Series([0] * len(row))
    row = np.where(row >= 0, row / pos_sum, -row / neg_sum)
    return row * scale 

def normalize_df(df, scale = 1):
    minutes = df.Minutes
    df = df.drop(columns=['Minutes'])
    df_norm = df.apply(normalize_row,scale = scale,axis=1)
    df_norm = pd.DataFrame(df_norm.tolist()) 
    df_norm.columns = df.columns
    return pd.concat([minutes, df_norm], axis=1)



def calculate_returns(alpha, close):
    dollar_amount = alpha.drop(columns=['Minutes'])
    closing_prices = close.drop(columns=['Minutes'])
    stock_returns = closing_prices.pct_change()
    weighted_returns = stock_returns * dollar_amount
    daily_returns = weighted_returns.sum(axis=1)
    return daily_returns

def sharpe_ratio(returns, risk_free_rate=0):
    mean_return = np.mean(returns)
    print(mean_return)
    std_return = np.std(returns)
    print(std_return)
    return (mean_return - risk_free_rate) / std_return

def sortino_ratio(returns, target_return=0):
    downside_returns = returns[returns <= target_return]
    downside_std = np.std(downside_returns)
    return (np.mean(returns) - target_return) / downside_std

def max_drawdown(returns):
    cumulative_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - peak) / peak
    return np.min(drawdowns) * 100

def max_returns(returns):
    return np.max(returns) * 100

def plot_cumulative_pnl_and_return(daily_returns):
    cumulative_pnl = np.cumsum(daily_returns)
    cumulative_return = np.cumprod(1 + daily_returns) - 1
    plt.plot(range(len(daily_returns)), cumulative_pnl, label='Cumulative PNL')
    plt.plot(range(len(daily_returns)), cumulative_return, label='Cumulative Return')
    plt.xlabel('Day')
    plt.ylabel('Cumulative')
    plt.legend()
    plt.title('Cumulative PNL vs Cumulative Return')
    plt.show()


def stats(returns, book_value):

    daily_returns_arr = np.array(list(map(lambda x: x / book_value, returns[1])))
    daily_returns = daily_returns_arr
    print(daily_returns)
    print("Sharpe Ratio:", sharpe_ratio(daily_returns_arr))
    print("Sortino Ratio:", sortino_ratio(daily_returns_arr))
    print("Max Drawdown:", max_drawdown(daily_returns_arr))
    print("Max Returns:", max_returns(daily_returns_arr))


    cumulative_pnl = np.cumsum(daily_returns)
    cumulative_return = np.cumprod(1 + daily_returns) - 1
    plt.plot(range(len(daily_returns)), cumulative_pnl, label='Cumulative PNL')
    plt.plot(range(len(daily_returns)), cumulative_return, label='Cumulative Return')
    plt.xlabel('Day')
    plt.ylabel('Cumulative')
    plt.legend()
    plt.title('Cumulative PNL vs Cumulative Return')
    plt.show()
