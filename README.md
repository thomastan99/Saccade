Given the time tight nature of the assignment, much of the developmental work was spent on understanding the data and getting out a framework for alpha and strategy generation.

The underlying ideas had some initial mean reversion/ momentum ideas which revolved around exploring the intraday momentum signals of individual tickers or mean reversion where it would go back to their average prices. These were standard strategies which form the baseline for most signals so I went ahead to try to start of with a 5 period mean reversion. 
To reduce exposure, I tried to implement a long-short neutral strategy by distributing weights equally across long and short positions to ensure that the strategy is market neutral, hence hedging on some risk in this aspect. In the functions.py file, I created several helper functions which were the baseline to creating signals. I tried to create code that was modularised which would make alpha creation a lot quicker as the framework as already in place. Additionally, I also used several statistics such as returns, drawdowns, sharpe and sortino ratios to evaluate the alphas. 
My intention was to create a framework where I could try out ideas, easily and then in the future could have some forms of hyperparameter turning introduced which can always be tested against the alpha statistics.

Unfortunately, I did not have enough time to complete the whole development process or go as far as I would have wished, but one used of machine learning I wanted to attempt before the deadline was actually to use un-supervised k-means clustering which would cluster stocks into different categories and then rank those categories before splitting them into the respective long/short positions. I also was able to have some iterations of trying to create alphas using the close/volume/open data but I didn’t have sufficient time to continue exploring. I hope that perhaps if possible I would have the opportunity to further discuss this project and ideas I had with the team to show case some of the other aspects of this project which I wanted to add in if I had more time.
