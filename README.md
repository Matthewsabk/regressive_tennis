# regressive_tennis

## Project Brief from codecademy.com

Provided in tennis_stats.csv is data from the men’s professional tennis league, which is called the ATP (Association of Tennis Professionals). Data from the top 1500 ranked players in the ATP over the span of 2009 to 2017 are provided in file. The statistics recorded for each player in each year include service game (offensive) statistics, return game (defensive) statistics and outcomes.

Create a linear regression model that predicts the outcome for a tennis player based on their playing habits. By analyzing and modeling the Association of Tennis Professionals (ATP) data,determine what it takes to be one of the best tennis players in the world.

## Data Overview
The dataset has shape: (1721, 24)
The tennis stats dataset contains columns: ['Player', 'Year', 'FirstServe', 'FirstServePointsWon',
       'FirstServeReturnPointsWon', 'SecondServePointsWon',
       'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted',
       'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved',
       'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon',
       'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon',
       'TotalPointsWon', 'TotalServicePointsWon', 'Wins', 'Losses', 'Winnings',
       'Ranking'],
The dataset includes statistics for the years 2009-2017 on 438 tennis players. This analysis primarily focused on 'Winnings' as the outcome feature.

## Analysis
Review scatter plots of features vs winnings to find potential linear relationships.
-- For loop iteration to visualize variables linear relationships.
Using sklearn, perform single variable linear regressions to determine which feature has the strongest linear relationship with 'Winnings'. 
-- 'single_linear_for_list' is a function to prepare the data, sample the data for a training and test set using sklearn.modelselection, and perform linear regression for each feature. It takes dataframe, a list of feature column names, and the outcome column names and returns a dictionary of the feature column and slope, intercept, and R2 score of the model for that features linear relationship with the outcome variable.
Using sklearn perform several two feature linear regressions to look for interactive relationships with Winnings.
-- 'twovar_regress' is a function that takes dataframe, outcome variable, feature 1 and feature 2 and returns the coefficients of the linear relationship model and the model's R2 score. It samples the values and displays a plot of the predicted vs actual Winnings as described by the model.
Using sklearn analyze multi-feature linear regressions with 'Winnings'
-- 'multi_var_regress' takes dataframe, outcome (str) and a features columns (list) and returns the coefficients of the linear relationship model and the model's R2 score
## Conclusions
There are linear relationships between 'Winnings' and the following features: '['Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved', 'DoubleFaults', 'ReturnGamesPlayed','ServiceGamesPlayed', 'Wins', 'Losses',]
Of these the strongest single feature relationships were with ReturnGamesPlayed and ServiceGamesPlayed, both of which had reasonable model scoring. This supports the unsurprising conclusion that players who play more games earn more in Winnings. Two of the models showed low model scoring and the relationships should not be considered those were with the features Aces and BreakPointsSaved. 

For the two feature models, the most robust models were for Wins & BreakPointOpportunities with an R2=0.862 and for ReturnGamesPlayed and ServiceGamesPlayed with an R2 or 0.867.

The final portion of the project is the multi-feature linear regression.
The features were divided into groups based on type of activity:
feature_group_1 = ['Aces','BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved','DoubleFaults']
feature_group_2 = ['ReturnGamesPlayed', 'ServiceGamesPlayed', 'Wins', 'Losses','Ranking']
feature_list_linear = ['Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved', 'DoubleFaults', 'ReturnGamesPlayed','ServiceGamesPlayed', 'Wins', 'Losses',]

The model scores produced were as follows:
feature_group_1 = 0.82
feature_group_2 = 0.87
feature_list_linear = 0.85

Final selection of features for inclusion in a multivariate linear regression to predict Winnings withing the tennis data set was based on the coefficients of the prior feature groups. The selected group was: ['BreakPointsOpportunities', 'BreakPointsSaved', 'Losses', 'Wins']. The model produced was the most accurate in this analysis, with an R2 score of 0.884.
