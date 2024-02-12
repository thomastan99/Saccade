import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)  
# pd.set_option('display.max_rows', None)     


from data_cleaning.dataCleaning import *

book_value = 1000000
daily_returns = initial_test_alpha(book_value)

