#libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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
#plt.savefig('hist_features_tennis.png')
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
#plt.savefig('scatter_feature_ranking.png')
plt.close('all')
#features vs winnings
posit = 1
plt.figure(3, figsize=[8,48])
for feature in tennis.columns:
    var = tennis[feature]
    plt.subplot(12,2,posit)
    plt.scatter(var, tennis['Winnings'])
    plt.title(f'{feature} vs Winnings')
    posit += 1
#plt.savefig('scatter_feature_winnings.png')
#plt.show()
plt.close('all')
"""Scatterplots suggest linear relationships between:
Ranking and Aces, BreakPointsFaced, BreakPointsOpportunities, DoubleFaults, ReturnGamesPlayed, ServiceGamesPlayed, Wins, Losses, and Winnings
Winnings and Aces, BreakPointsFaced, BreakPointsOpportunities, DoubleFaults, ReturnGamesPlayed, Wins, Losses, and Ranking
"""
## perform single feature linear regressions here:
#Split data into training and test sets for regression model testing.
aces = tennis[['Aces']]
winnings= tennis[['Winnings']]
reshape_aces = np.array(aces).reshape(-1,1)
scaler = StandardScaler()
aces_scaled= scaler.fit_transform(reshape_aces)
reshape_winnings = np.array(winnings).reshape(-1,1)
winnings_scaled = scaler.fit_transform(reshape_winnings)
aces_train, aces_test, winnings_train, winnings_test = tts(aces_scaled, winnings_scaled, train_size=0.8, test_size=0.2, random_state = 145)
#create model of aces vs winnings
aces_winnings_regr = LinearRegression()
aces_winnings_regr.fit(aces_train, winnings_train)
predict_winnings_aces = aces_winnings_regr.predict(aces_test)
aces_winnings_model_score = aces_winnings_regr.score(aces_test, winnings_test)
plt.figure(4, figsize=(6,5))
plt.scatter(aces_test, winnings_test)
plt.plot(aces_test, predict_winnings_aces)
plt.title("Aces vs Ranking")
#plt.savefig('aces_ranking__scaled_best_fit.png')
#plt.show()
plt.close('all')

#create a function to run through the possible relationships. 
def single_linear_for_list(df, col_list, outcome_col):
    single_var_linear_regr = {}
    y_col = df[[outcome_col]]
    #standardize outcome
    def standardize(df_col):
        scaler = StandardScaler()
        reshape_var = np.array(df_col).reshape(-1,1)
        var_scaled = scaler.fit_transform(reshape_var)
        return var_scaled
    y= standardize(y_col)
    #for each of the possible linear relationships
    for col in col_list:
        x_col = df[[col]]
        #standardize and normalize feature
        x= standardize(x_col)
        #split the data into train/test sets
        x_train, x_test, y_train, y_test = tts(x,y, train_size=0.8, test_size=0.2, random_state=5502)
        #create the model
        xy_regr = LinearRegression()
        xy_regr.fit(x_train, y_train)
        #predict the line of best fit
        predict_y = xy_regr.predict(x_test)
        #coefficients and intercept
        m = xy_regr.coef_
        b = xy_regr.intercept_
        #score model
        r2 = xy_regr.score(x_test, y_test)
        #plot the scatter of the data and the line of best fit
        plt.figure(figsize=(6,5))
        plt.scatter(x_test, y_test, alpha=0.4)
        plt.plot(x_test, predict_y, color ='r')
        plt.title(f"{col} vs {outcome_col}")
        #plt.savefig(f'{col.lower()}_{outcome_col.lower()}_best_fit.png')
        #save line of best fit:
        single_var_linear_regr[col]= {'m':m, 'b':b, 'r2':r2}
        #if r2> 0.7:
            #plt.show()
        plt.close('all')
    return single_var_linear_regr

#Use linear regression function to create linear regressions for all variables with linear relationship scatterplot
feature_list_linear = ['Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved',
       'DoubleFaults', 'ReturnGamesPlayed','ServiceGamesPlayed', 'Wins', 'Losses',]
#generate single feature linear regressions for winnings
coeff_linear_relationships_winnings = single_linear_for_list(tennis, feature_list_linear, "Winnings")
df_winnings_coef = pd.DataFrame(coeff_linear_relationships_winnings)
print(df_winnings_coef)

## perform two feature linear regressions here:
def twovar_regress(df, target, var1, var2):
    y, x1, x2 = target, var1, var2
    X = df[[x1, x2]]
    y_actual = df[[y]]
    x_train, x_test, y_train, y_test =tts(X,y_actual, train_size=0.8, test_size=0.2)
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)
    y_predict_x1_x2 = mlr.predict(x_test)
    model_score = mlr.score(x_test,y_test)
    coeff_mlr = mlr.coef_
    print(f"Score of modeled winnings based on {str.title(x1)} and {str.title(x2)}: {model_score}. Coefficents are: {coeff_mlr}")
    plt.figure(5, figsize=(5,5))
    plt.scatter(y_test, y_predict_x1_x2, alpha=0.4)
    plt.title(f"{str.title(y)} actual vs predicted based on {str.title(x1)} and {str.title(x2)}")
    plt.xlabel(f"{str.title(y)} Actual")
    plt.ylabel(f"Predicted {str.title(y)}")
    #plt.show()
    plt.close('all')
    return coeff_mlr, model_score


twovar_regress(tennis, 'Winnings','Wins', 'BreakPointsOpportunities')
twovar_regress(tennis, 'Winnings','DoubleFaults', 'BreakPointsFaced')
twovar_regress(tennis, 'Winnings','Wins', 'Losses')
twovar_regress(tennis, 'Winnings','ReturnGamesPlayed', 'ServiceGamesPlayed')

## perform multiple feature linear regressions here:
##group1 is servepoints, group2 is breakpoints, group 3 is return vs service games, and 4 is wins/losses/ranking
feature_group_1 = ['Aces','BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved','DoubleFaults']
feature_group_2 = ['ReturnGamesPlayed', 'ServiceGamesPlayed', 'Wins', 'Losses','Ranking']
feature_list_linear = ['Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved', 'DoubleFaults', 'ReturnGamesPlayed','ServiceGamesPlayed', 'Wins', 'Losses',]
#create function to run linear regressions
def multi_var_regress(df, outcome, feature_list):
    y = outcome
    X = df[feature_list]
    y_actual = df[[y]]
    x_train, x_test, y_train, y_test =tts(X,y_actual, train_size=0.8, test_size=0.2)
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)
    print (mlr.coef_.round())
    y_predict_x1_x2 = mlr.predict(x_test)
    model_score = mlr.score(x_test,y_test)
    print(f"Score of modeled winnings based on {feature_list}: {model_score.round(3)}")
    return mlr.coef_.round(), model_score.round(3)

#run the linear regressions on the feature sets as needed
multi_var_regress (tennis, 'Winnings', feature_group_1)
multi_var_regress (tennis, 'Winnings', feature_group_2)
multi_var_regress (tennis, 'Winnings', feature_list_linear)

#select a feature group based on coefficients in other groups and run a multivariate linear regression. 
selected_feature_group = ['BreakPointsOpportunities', 'BreakPointsSaved', 'Losses','Wins']
multi_var_regress (tennis, 'Winnings', selected_feature_group)