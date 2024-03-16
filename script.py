#libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# set display preferences:
pd.set_option('display.max_columns', 500)

# load and investigate the data here:
tennis = pd.read_csv("tennis_stats.csv")
features = tennis.columns
print(f"The dataset has shape: {tennis.shape}")
print(f"The tennis stats dataset contains columns: {features}")
print(f"Datatypes: {tennis.dtypes}")
print(tennis.describe(include='all'))
print(f"Datapoints per year {tennis.Year.value_counts().sort_values()}")
print(tennis.nunique())
print("Possible outcome variables to focus on include Wins, Winnings, and Ranking")
print(f"Sum of wins: {tennis.Wins.sum()}. Sum of losses: {tennis.Losses.sum()}")

# perform exploratory analysis here:
# visualize data
posit = 1
plt.figure(1, figsize=[8,48])
for feature in tennis.columns:
    var = tennis[feature]
    plt.subplot(12,2,posit)
    plt.hist(var)
    plt.title(f'{feature}')
    posit += 1
plt.savefig('hist_features_tennis.png')
plt.close('all')
"""Note on features:
Aces is right-skewed
BreakPointsConverted has outliers
BreakPointsFaced, BreakPointsOpportunities, DoubleFaults, ReturnGamesPlayed, ServiceGamesPlayed, Wins, Losses, and Winnings are right-skewed
"""
#drop Player from potential relationships
tennis.drop(columns='Player', inplace=True)
#visualize potential relationships
#features vs ranking
posit = 1
plt.figure(2, figsize=[8,48])
for feature in tennis.columns:
    var = tennis[feature]
    plt.subplot(12,2,posit)
    plt.scatter(var, tennis['Ranking'])
    plt.title(f'{feature} vs Ranking')
    posit += 1
plt.savefig('scatter_feature_ranking.png')
plt.close('all')
#winnings vs ranking
posit = 1
plt.figure(3, figsize=[8,48])
for feature in tennis.columns:
    var = tennis[feature]
    plt.subplot(12,2,posit)
    plt.scatter(var, tennis['Winnings'])
    plt.title(f'{feature} vs Winnings')
    posit += 1
plt.savefig('scatter_feature_winnings.png')
plt.close('all')
"""Scatterplots suggest linear relationships between:
Ranking and Aces, BreakPointsFaced, BreakPointsOpportunities, DoubleFaults, ReturnGamesPlayed, Service Games Played, Wins, Losses, and Winnings
Winnings and Aces, BreakPointsFaced, BreakPointsOpportunities, DoubleFaults, ReturnGamesPlayed, Wins, Losses, and Ranking
"""
## perform single feature linear regressions here:






















## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:
