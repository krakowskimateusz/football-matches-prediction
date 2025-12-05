import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('soccerData.csv')
df['date']= pd.to_datetime(df['date'])
df['season_start_year'] = df['date'].apply(lambda d: d.year if d.month >= 8 else d.year - 1)
df['season'] = df['season_start_year'].astype(str) + '/' + (df['season_start_year'] + 1).astype(str)

raw_match_stats = df[[
                'date',
                'match_id',
                'home_team_name',
                'away_team_name',
                'home_team_goal_count', 
                'away_team_goal_count',
                'home_team_half_time_goal_count',
                'away_team_half_time_goal_count',
                'home_team_shots',
                'away_team_shots',
                'home_team_shots_on_target',
                'away_team_shots_on_target',
                'home_team_fouls',
                'away_team_fouls',
                'home_team_corner_count',
                'away_team_corner_count',
                'home_team_yellow',
                'away_team_yellow',
                'home_team_red',
                'away_team_red'
                ]]


raw_match_stats.sort_values(by=['date'], ascending=False, inplace=True)
raw_match_stats.dropna(inplace=True)

raw_match_stats.loc[raw_match_stats['home_team_goal_count'] == raw_match_stats['away_team_goal_count'], 'home_team_result'] = 1
raw_match_stats.loc[raw_match_stats['home_team_goal_count'] > raw_match_stats['away_team_goal_count'], 'home_team_result'] = 3
raw_match_stats.loc[raw_match_stats['home_team_goal_count'] < raw_match_stats['away_team_goal_count'], 'home_team_result'] = 0

raw_match_stats.loc[raw_match_stats['home_team_goal_count'] == raw_match_stats['away_team_goal_count'], 'away_team_result'] = 1
raw_match_stats.loc[raw_match_stats['home_team_goal_count'] > raw_match_stats['away_team_goal_count'], 'away_team_result'] = 0
raw_match_stats.loc[raw_match_stats['home_team_goal_count'] < raw_match_stats['away_team_goal_count'], 'away_team_result'] = 3


# Split the raw_match_stats to two datasets (home_team_stats and away_team_stats)
home_team_stats = raw_match_stats[[
 'date',
 'match_id',
 'home_team_name',
 'home_team_goal_count',
 'home_team_half_time_goal_count',
 'home_team_corner_count',
 'home_team_shots',
 'home_team_shots_on_target',
 'home_team_fouls',
 'home_team_yellow',
 'home_team_red',
 'home_team_result',
 'away_team_goal_count',
 'away_team_half_time_goal_count',
 'away_team_corner_count',
 'away_team_shots',
 'away_team_shots_on_target',
 'away_team_fouls',
 'away_team_yellow',
 'away_team_red']]

home_team_stats = home_team_stats.rename(columns={'home_team_name':'name',
                                                'home_team_goal_count':'goalsScored',
                                                'home_team_half_time_goal_count':'halfTimeGoalsScored',
                                                'home_team_corner_count':'cornerCount',
                                                'home_team_shots':'shots',
                                                'home_team_shots_on_target':'shotsOnTarget',
                                                'home_team_fouls':'foulsConceded',
                                                'home_team_yellow':'yellowConceded',
                                                'home_team_red':'redConceded',
                                                'home_team_result':'result',
                                                'away_team_goal_count':'goalsConceded',
                                                'away_team_half_time_goal_count':'halfTimeGoalsConceded',
                                                'away_team_corner_count':'cornersConceded',
                                                'away_team_shots':'shotsConceded',
                                                'away_team_shots_on_target':'shotsOnTargetConceded',
                                                'away_team_fouls':'foulsReceived',
                                                'away_team_yellow':'yellowOpponent',
                                                'away_team_red':'redOpponent'})

away_team_stats = raw_match_stats[[
 'date',
 'match_id',
 'away_team_name',
 'away_team_goal_count',
 'away_team_half_time_goal_count',
 'away_team_corner_count',
 'away_team_shots',
 'away_team_shots_on_target',
 'away_team_fouls',
 'away_team_yellow',
 'away_team_red',
 'away_team_result',
 'home_team_goal_count',
 'home_team_half_time_goal_count',
 'home_team_corner_count',
 'home_team_shots',
 'home_team_shots_on_target',
 'home_team_fouls',
 'home_team_yellow',
 'home_team_red',]]

away_team_stats = away_team_stats.rename(columns={'away_team_name':'name',
                                                'away_team_goal_count':'goalsScored',
                                                'away_team_half_time_goal_count':'halfTimeGoalsScored',
                                                'away_team_corner_count':'cornerCount',
                                                'away_team_shots':'shots',
                                                'away_team_shots_on_target':'shotsOnTarget',
                                                'away_team_fouls':'foulsConceded',
                                                'away_team_yellow':'yellowConceded',
                                                'away_team_red':'redConceded',
                                                'away_team_result':'result',
                                                'home_team_goal_count':'goalsConceded',
                                                'home_team_half_time_goal_count':'halfTimeGoalsConceded',
                                                'home_team_corner_count':'cornersConceded',
                                                'home_team_shots':'shotsConceded',
                                                'home_team_shots_on_target':'shotsOnTargetConceded',
                                                'home_team_fouls':'foulsReceived',
                                                'home_team_yellow':'yellowOpponent',
                                                'home_team_red':'redOpponent'})

# add an additional column to denote whether the team is playing at home or away - this will help us later
home_team_stats['home_or_away']='Home'
away_team_stats['home_or_away']='Away'

# stack these two datasets so that each row is the stats for a team for one match (team_stats_per_match)
team_stats_per_match = pd.concat([home_team_stats,away_team_stats])

# Podgląd
# team_stats_per_match.to_csv('team_stats_per_match.csv', index=False)


avg_lastFive_stat_columns = [
                    'average_goalsScored_last_five',
                    'average_halfTimeGoalsScored_last_five',
                    'average_cornerCount_last_five',
                    'average_shots_last_five',
                    'average_shotsOnTarget_last_five',
                    'average_foulsConceded_last_five',
                    'average_yellowConceded_last_five',
                    'average_redConceded_last_five',
                    'average_result_last_five',
                    'average_goalsConceded_last_five',
                    'average_halfTimeGoalsConceded_last_five',
                    'average_cornersConceded_last_five',
                    'average_shotsConceded_last_five',
                    'average_shotsOnTargetConceded_last_five',
                    'average_foulsReceived_last_five',
                    'average_yellowOpponent_last_five',
                    'average_redOpponent_last_five'
                    ]

lastFive_stats_list = []
for index, row in team_stats_per_match.iterrows():
    team_stats_last_five_matches = team_stats_per_match.loc[
        (team_stats_per_match['name'] == row['name']) & 
        (team_stats_per_match['date'] < row['date'])
    ].sort_values(by=['date'], ascending=False)
    
    lastFive_stats_list.append(
        team_stats_last_five_matches.iloc[0:5, 3:-1].mean(axis=0).values[0:18]
    )

avg_lastFive_stats_per_team = pd.DataFrame(
    lastFive_stats_list, 
    columns=avg_lastFive_stat_columns
)

avg_lastFive_stats_per_team.index = team_stats_per_match.index

team_stats_with_averages = pd.concat(
    [team_stats_per_match, avg_lastFive_stats_per_team], 
    axis=1
)



seasons = sorted(team_stats_with_averages['season'].unique())
teams_by_season = (
    team_stats_with_averages.groupby('season')['name']
    .unique()
    .apply(set)
    .to_dict()
)

promoted_dict = {}

promoted_dict['1999/2000'] = set(['Sunderland', 'Bradford City', 'Watford'])

for i in range(1, len(seasons)):
    current_season = seasons[i]
    prev_season = seasons[i - 1]
    current_teams = teams_by_season.get(current_season, set())
    prev_teams = teams_by_season.get(prev_season, set())
    promoted_teams = list(current_teams - prev_teams)
    promoted_dict[current_season] = set(promoted_teams)

team_stats_with_averages['promoted'] = team_stats_with_averages.apply(
    lambda row: int(row['name'] in promoted_dict.get(row['season'], set())),
    axis=1
)


team_stats_with_averages['not_lost'] = (team_stats_with_averages['result'] >= 1).astype(int)
team_stats_with_averages['win'] = (team_stats_with_averages['result'] == 3).astype(int)

def calc_streaks(team_df):
    team_df = team_df.sort_values('date')
    
    win_streak = 0
    unbeaten_streak = 0
    
    win_streaks = []
    unbeaten_streaks = []
    
    for res in team_df['result']:
        win_streaks.append(win_streak)
        unbeaten_streaks.append(unbeaten_streak)
        
        if res == 3:
            win_streak += 1
            unbeaten_streak += 1
        elif res == 1:
            win_streak = 0
            unbeaten_streak += 1
        else:
            win_streak = 0
            unbeaten_streak = 0
    
    team_df['win_streak_before_match'] = win_streaks
    team_df['unbeaten_streak_before_match'] = unbeaten_streaks
    
    return team_df

team_stats_with_averages = team_stats_with_averages.groupby('name', group_keys=False).apply(calc_streaks)

team_stats_with_averages = team_stats_with_averages.sort_values('date').reset_index(drop=True)



team_stats_with_averages.to_csv('team_stats_with_averages.csv', index=False)






# # At each row of this dataset, get the team name, find the stats for that team during the last 5 matches, and average these stats (avg_stats_per_team). 
# avg_lastTen_stat_columns = [
#                     'average_goalsScored_last_ten',
#                     'average_halfTimeGoalsScored_last_ten',
#                     'average_cornerCount_last_ten',
#                     'average_shots_last_ten',
#                     'average_shotsOnTarget_last_ten',
#                     'average_foulsConceded_last_ten',
#                     'average_yellowConceded_last_ten',
#                     'average_redConceded_last_ten',
#                     'average_result_last_ten',
#                     'average_goalsConceded_last_ten',
#                     'average_halfTimeGoalsConceded_last_ten',
#                     'average_cornersConceded_last_ten',
#                     'average_shotsConceded_last_ten',
#                     'average_shotsOnTargetConceded_last_ten',
#                     'average_foulsReceived_last_ten',
#                     'average_yellowOpponent_last_ten',
#                     'average_redOpponent_last_ten'
#                     ]

# lastTen_stats_list = []
# for index, row in team_stats_per_match.iterrows():
#     team_stats_last_ten_matches = team_stats_per_match.loc[(team_stats_per_match['name']==row['name']) & (team_stats_per_match['date']<row['date'])].sort_values(by=['date'], ascending=False)
#     lastTen_stats_list.append(team_stats_last_ten_matches.iloc[0:10,3:-1].mean(axis=0).values[0:18])

# avg_lastTen_stats_per_team = pd.DataFrame(lastTen_stats_list, columns=avg_lastTen_stat_columns)


# avg_lastFiveHome_stat_columns=[
#                     'average_goalsScored_last_five_home',
#                     'average_halfTimeGoalsScored_last_five_home',
#                     'average_cornerCount_last_five_home',
#                     'average_shots_last_five_home',
#                     'average_shotsOnTarget_last_five_home',
#                     'average_foulsConceded_last_five_home',
#                     'average_yellowConceded_last_five_home',
#                     'average_redConceded_last_five_home',
#                     'average_result_last_five_home',
#                     'average_goalsConceded_last_five_home',
#                     'average_halfTimeGoalsConceded_last_five_home',
#                     'average_cornersConceded_last_five_home',
#                     'average_shotsConceded_last_five_home',
#                     'average_shotsOnTargetConceded_last_five_home',
#                     'average_foulsReceived_last_five_home',
#                     'average_yellowOpponent_last_five_home',
#                     'average_redOpponent_last_five_home'
#                     ]

# lastFive_Home_stats_list = []
# team_stats_L5_home_matches = team_stats_per_match[team_stats_per_match['home_or_away'] == 'Home']
# for index, row in team_stats_L5_home_matches.iterrows():
#     team_stats_last_five_home_matches = team_stats_L5_home_matches.loc[(team_stats_L5_home_matches['name']==row['name']) & (team_stats_L5_home_matches['date']<row['date'])].sort_values(by=['date'], ascending=False)
#     lastFive_Home_stats_list.append(team_stats_last_five_home_matches.iloc[0:5,3:-1].mean(axis=0).values[0:18])

# avg_lastFiveHome_stats_per_team = pd.DataFrame(lastFive_Home_stats_list, columns=avg_lastFiveHome_stat_columns)

# team_stats_L5_home_matches = pd.concat([team_stats_L5_home_matches.reset_index(drop=True), avg_lastFiveHome_stats_per_team], axis=1, ignore_index=False)

# avg_lastFiveAway_stat_columns=[
#                     'average_goalsScored_last_five_away',
#                     'average_halfTimeGoalsScored_last_five_away',
#                     'average_cornerCount_last_five_away',
#                     'average_shots_last_five_away',
#                     'average_shotsOnTarget_last_five_away',
#                     'average_foulsConceded_last_five_away',
#                     'average_yellowConceded_last_five_away',
#                     'average_redConceded_last_five_away',
#                     'average_result_last_five_away',
#                     'average_goalsConceded_last_five_away',
#                     'average_halfTimeGoalsConceded_last_five_away',
#                     'average_cornersConceded_last_five_away',
#                     'average_shotsConceded_last_five_away',
#                     'average_shotsOnTargetConceded_last_five_away',
#                     'average_foulsReceived_last_five_away',
#                     'average_yellowOpponent_last_five_away',
#                     'average_redOpponent_last_five_away'
#                     ]

# lastFive_away_stats_list = []
# team_stats_L5_away_matches = team_stats_per_match[team_stats_per_match['home_or_away'] == 'Away']
# for index, row in team_stats_L5_away_matches.iterrows():
#     team_stats_last_five_away_matches = team_stats_L5_away_matches.loc[(team_stats_L5_away_matches['name']==row['name']) & (team_stats_L5_away_matches['date']<row['date'])].sort_values(by=['date'], ascending=False)
#     lastFive_away_stats_list.append(team_stats_last_five_away_matches.iloc[0:5,3:-1].mean(axis=0).values[0:18])

# avg_lastFiveAway_stats_per_team = pd.DataFrame(lastFive_away_stats_list, columns=avg_lastFiveAway_stat_columns)
# team_stats_L5_away_matches = pd.concat([team_stats_L5_away_matches.reset_index(drop=True), avg_lastFiveAway_stats_per_team], axis=1, ignore_index=False)

# team_stats_L5_home_matches.columns = team_stats_L5_home_matches.columns[:2].tolist() + ['team_1_'+str(col) for col in team_stats_L5_home_matches.columns[2:]]
# team_stats_L5_away_matches.columns = team_stats_L5_away_matches.columns[:2].tolist() + ['team_2_'+str(col) for col in team_stats_L5_away_matches.columns[2:]]

# home_and_away_stats = pd.merge(team_stats_L5_home_matches,team_stats_L5_away_matches,how='left',on=['date','match_id'])


# team_stats_per_match = pd.concat([team_stats_per_match.reset_index(drop=True), avg_lastTen_stats_per_team], axis=1, ignore_index=False)
# # Re-segment the home and away teams.
# home_team_stats = team_stats_per_match.iloc[:int(team_stats_per_match.shape[0]/2),:]
# away_team_stats = team_stats_per_match.iloc[int(team_stats_per_match.shape[0]/2):,:]

# home_team_stats.columns = home_team_stats.columns[:2].tolist() + ['team_1_'+str(col) for col in home_team_stats.columns[2:]]
# away_team_stats.columns = away_team_stats.columns[:2].tolist() + ['team_2_'+str(col) for col in away_team_stats.columns[2:]]

# # Combine at each match to get a dataset with a row representing each match. 
# # drop the NA rows (earliest match for each team, i.e no previous stats)
# away_team_stats = away_team_stats.iloc[:, 2:]
# match_stats = pd.concat([home_team_stats, away_team_stats.reset_index(drop=True)], axis=1, ignore_index=False)
# match_stats = match_stats.dropna().reset_index(drop=True)

# match_stats=pd.merge(match_stats,home_and_away_stats,how='left',on=['date',
#                                                                     'match_id',
#                                                                     'team_1_name',
#                                                                     'team_1_goalsScored',
#                                                                     'team_1_halfTimeGoalsScored',
#                                                                     'team_1_cornerCount',
#                                                                     'team_1_shots',
#                                                                     'team_1_shotsOnTarget',
#                                                                     'team_1_foulsConceded',
#                                                                     'team_1_yellowConceded',
#                                                                     'team_1_redConceded',
#                                                                     'team_1_result',
#                                                                     'team_1_goalsConceded',
#                                                                     'team_1_halfTimeGoalsConceded',
#                                                                     'team_1_cornersConceded',
#                                                                     'team_1_shotsConceded',
#                                                                     'team_1_shotsOnTargetConceded',
#                                                                     'team_1_foulsReceived',
#                                                                     'team_1_yellowOpponent',
#                                                                     'team_1_redOpponent',
#                                                                     'team_1_home_or_away',
#                                                                     'team_2_name',
#                                                                     'team_2_goalsScored',
#                                                                     'team_2_halfTimeGoalsScored',
#                                                                     'team_2_cornerCount',
#                                                                     'team_2_shots',
#                                                                     'team_2_shotsOnTarget',
#                                                                     'team_2_foulsConceded',
#                                                                     'team_2_yellowConceded',
#                                                                     'team_2_redConceded',
#                                                                     'team_2_result',
#                                                                     'team_2_goalsConceded',
#                                                                     'team_2_halfTimeGoalsConceded',
#                                                                     'team_2_cornersConceded',
#                                                                     'team_2_shotsConceded',
#                                                                     'team_2_shotsOnTargetConceded',
#                                                                     'team_2_foulsReceived',
#                                                                     'team_2_yellowOpponent',
#                                                                     'team_2_redOpponent',
#                                                                     'team_2_home_or_away'])


# match_stats.dropna(inplace=True)

# # Define features
# features = ['team_1_average_goalsScored_last_ten',
#             'team_1_average_halfTimeGoalsScored_last_ten',
#             'team_1_average_cornerCount_last_ten',
#             'team_1_average_shots_last_ten',
#             'team_1_average_shotsOnTarget_last_ten',
#             'team_1_average_foulsConceded_last_ten',
#             'team_1_average_yellowConceded_last_ten',
#             'team_1_average_redConceded_last_ten',
#             'team_1_average_result_last_ten',
#             'team_1_average_goalsConceded_last_ten',
#             'team_1_average_halfTimeGoalsConceded_last_ten',
#             'team_1_average_cornersConceded_last_ten',
#             'team_1_average_shotsConceded_last_ten',
#             'team_1_average_shotsOnTargetConceded_last_ten',
#             'team_1_average_foulsReceived_last_ten',
#             'team_1_average_yellowOpponent_last_ten',
#             'team_1_average_redOpponent_last_ten',
#             'team_2_average_goalsScored_last_ten',
#             'team_2_average_halfTimeGoalsScored_last_ten',
#             'team_2_average_cornerCount_last_ten',
#             'team_2_average_shots_last_ten',
#             'team_2_average_shotsOnTarget_last_ten',
#             'team_2_average_foulsConceded_last_ten',
#             'team_2_average_yellowConceded_last_ten',
#             'team_2_average_redConceded_last_ten',
#             'team_2_average_result_last_ten',
#             'team_2_average_goalsConceded_last_ten',
#             'team_2_average_halfTimeGoalsConceded_last_ten',
#             'team_2_average_cornersConceded_last_ten',
#             'team_2_average_shotsConceded_last_ten',
#             'team_2_average_shotsOnTargetConceded_last_ten',
#             'team_2_average_foulsReceived_last_ten',
#             'team_2_average_yellowOpponent_last_ten',
#             'team_2_average_redOpponent_last_ten',
#             'team_1_average_goalsScored_last_five_home',
#             'team_1_average_halfTimeGoalsScored_last_five_home',
#             'team_1_average_cornerCount_last_five_home',
#             'team_1_average_shots_last_five_home',
#             'team_1_average_shotsOnTarget_last_five_home',
#             'team_1_average_foulsConceded_last_five_home',
#             'team_1_average_yellowConceded_last_five_home',
#             'team_1_average_redConceded_last_five_home',
#             'team_1_average_result_last_five_home',
#             'team_1_average_goalsConceded_last_five_home',
#             'team_1_average_halfTimeGoalsConceded_last_five_home',
#             'team_1_average_cornersConceded_last_five_home',
#             'team_1_average_shotsConceded_last_five_home',
#             'team_1_average_shotsOnTargetConceded_last_five_home',
#             'team_1_average_foulsReceived_last_five_home',
#             'team_1_average_yellowOpponent_last_five_home',
#             'team_1_average_redOpponent_last_five_home',
#             'team_2_average_goalsScored_last_five_away',
#             'team_2_average_halfTimeGoalsScored_last_five_away',
#             'team_2_average_cornerCount_last_five_away',
#             'team_2_average_shots_last_five_away',
#             'team_2_average_shotsOnTarget_last_five_away',
#             'team_2_average_foulsConceded_last_five_away',
#             'team_2_average_yellowConceded_last_five_away',
#             'team_2_average_redConceded_last_five_away',
#             'team_2_average_result_last_five_away',
#             'team_2_average_goalsConceded_last_five_away',
#             'team_2_average_halfTimeGoalsConceded_last_five_away',
#             'team_2_average_cornersConceded_last_five_away',
#             'team_2_average_shotsConceded_last_five_away',
#             'team_2_average_shotsOnTargetConceded_last_five_away',
#             'team_2_average_foulsReceived_last_five_away',
#             'team_2_average_yellowOpponent_last_five_away',
#             'team_2_average_redOpponent_last_five_away'
# ]


















# train_data = match_stats[match_stats['date'] < '2018-07-01']
# test_data = match_stats[match_stats['date'] >= '2018-07-01']
# test_data = test_data[test_data['date'] < '2023-10-01']
# upcoming_matches = match_stats[match_stats['date'] >= '2023-10-01']

# X_train = train_data[features]
# X_test = test_data[features]
# Y_train_team1 = train_data['team_1_goalsScored']
# Y_test_team1 = test_data['team_1_goalsScored']
# Y_train_team2 = train_data['team_2_goalsScored']
# Y_test_team2 = test_data['team_2_goalsScored']


# import pickle
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression

# from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor
# from xgboost import XGBRegressor
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline

# # Nazwy i modele regresyjne
# names = [
#     "Linear Regression",
#     "Random Forest",
#     "Decision Tree",
#     "Support Vector Regressor",
#     "XGBoost",
#     "Neural Network",
# ]

# regressors = [
#     LinearRegression(),
#     RandomForestRegressor(max_depth=5, n_estimators=100, max_features='sqrt', random_state=42),
#     DecisionTreeRegressor(max_depth=6, random_state=42),
#     make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.1)),
#     XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42),
#     make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
# ]


# # Define file paths for saving the models
# model_save_paths = {name: f'{name}_model.pkl' for name in names}

# # Train models and save them as pickle files for team_1_goalsScored
# for name, reg in zip(names, regressors):
#     # Fit the regressor on the training data for team_1_goalsScored
#     reg.fit(X_train, Y_train_team1)

#     # Save the model to a pickle file
#     with open(model_save_paths[name] + '_team1', 'wb') as f:
#         pickle.dump(reg, f)

#     # Make predictions on the test data
#     test_data[name + '_team_1_goalsScored'] = reg.predict(X_test)
#     mse_team1 = mean_squared_error(Y_test_team1, test_data[name + '_team_1_goalsScored'])
#     r2_team1 = r2_score(Y_test_team1, test_data[name + '_team_1_goalsScored'])

#     # Print evaluation metrics for each regressor
#     print(f'{name} Team 1 GoalsScored MSE: {mse_team1}')
#     print(f'{name} Team 1 GoalsScored R^2: {r2_team1}')

# # Train models and save them as pickle files for team_2_goalsScored
# for name, reg in zip(names, regressors):
#     # Fit the regressor on the training data for team_2_goalsScored
#     reg.fit(X_train, Y_train_team2)

#     # Save the model to a pickle file
#     with open(model_save_paths[name] + '_team2', 'wb') as f:
#         pickle.dump(reg, f)

#     # Make predictions on the test data
#     test_data[name + '_team_2_goalsScored'] = reg.predict(X_test)
#     mse_team2 = mean_squared_error(Y_test_team2, test_data[name + '_team_2_goalsScored'])
#     r2_team2 = r2_score(Y_test_team2, test_data[name + '_team_2_goalsScored'])

#     # Print evaluation metrics for each regressor
#     print(f'{name} Team 2 GoalsScored MSE: {mse_team2}')
#     print(f'{name} Team 2 GoalsScored R^2: {r2_team2}')

# for name in names:
#     # wczytaj zapisane modele
#     with open(model_save_paths[name] + '_team1', 'rb') as f:
#         mdl1 = pickle.load(f)
#     with open(model_save_paths[name] + '_team2', 'rb') as f:
#         mdl2 = pickle.load(f)

#     # predykcje na train
#     p1_train = mdl1.predict(X_train)
#     p2_train = mdl2.predict(X_train)
#     pred_outcome_train = np.where(np.isclose(p1_train, p2_train, atol=0.2), 1, np.where(p1_train > p2_train, 3, 0))
#     true_outcome_train = train_data['team_1_result'].values
#     try:
#         train_acc = np.mean(pred_outcome_train == true_outcome_train)
#     except Exception:
#         train_acc = np.nan

#     # predykcje na test
#     p1_test = mdl1.predict(X_test)
#     p2_test = mdl2.predict(X_test)
#     pred_outcome_test = np.where(np.isclose(p1_test, p2_test, atol=0.2), 1, np.where(p1_test > p2_test, 3, 0))
#     true_outcome_test = test_data['team_1_result'].values
#     mask = ~np.isnan(true_outcome_test)
#     if mask.sum() == 0:
#         test_acc = np.nan
#     else:
#         test_acc = np.mean(pred_outcome_test[mask] == true_outcome_test[mask])

#     print(f'{name} train_outcome_acc: {train_acc:.4f} | test_outcome_acc: {test_acc:.4f}')

# # Apply the saved models to the 'upcoming_matches' dataframe for team_1_goalsScored and team_2_goalsScored
# for name, reg in zip(names, regressors):
#     # Load the models from the pickle files
#     with open(model_save_paths[name] + '_team1', 'rb') as f:
#         loaded_model_team1 = pickle.load(f)
#     with open(model_save_paths[name] + '_team2', 'rb') as f:
#         loaded_model_team2 = pickle.load(f)

#     # Predict team_1_goalsScored and team_2_goalsScored for upcoming matches
#     upcoming_matches[name + '_team_1_goalsScored'] = loaded_model_team1.predict(upcoming_matches[features])
#     upcoming_matches[name + '_team_2_goalsScored'] = loaded_model_team2.predict(upcoming_matches[features])




# # Export the predictions for upcoming matches to a CSV file
# upcoming_matches=upcoming_matches[['date',
#                                     'match_id',
#                                     'team_1_name',
#                                     'team_2_name',
#                                     'Linear Regression_team_1_goalsScored',
#                                     'Linear Regression_team_2_goalsScored',
#                                     'Random Forest_team_1_goalsScored',
#                                     'Random Forest_team_2_goalsScored',
#                                     'Decision Tree_team_1_goalsScored',
#                                     'Decision Tree_team_2_goalsScored',
#                                     'Support Vector Regressor_team_1_goalsScored',
#                                     'Support Vector Regressor_team_2_goalsScored',
#                                     'XGBoost_team_1_goalsScored',
#                                     'XGBoost_team_2_goalsScored',
#                                     'Neural Network_team_1_goalsScored',
#                                     'Neural Network_team_2_goalsScored'
#                                     ]]
# upcoming_matches.to_csv('upcoming_matches_goals_predictions.csv', index=False)


# # Evaluate upcoming_matches predictions by joining true results from match_stats (match_id must match)
# preds = upcoming_matches.copy()

# # bring true results from match_stats
# truth_cols = ['match_id','team_1_goalsScored','team_2_goalsScored','team_1_result']
# truth = match_stats[truth_cols].drop_duplicates(subset='match_id')
# preds = preds.merge(truth, on='match_id', how='left', suffixes=('','_true'))

# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# eval_rows = []
# for name in names:
#     p1 = preds[f'{name}_team_1_goalsScored'].values
#     p2 = preds[f'{name}_team_2_goalsScored'].values
#     t1 = preds['team_1_goalsScored'].values
#     t2 = preds['team_2_goalsScored'].values

#     mask = ~np.isnan(t1) 
#     if mask.sum() == 0:
#         print(f'Brak prawdziwych wyników dla modelu {name}')
#         continue

#     p1m, p2m, t1m, t2m = p1[mask], p2[mask], t1[mask], t2[mask]

#     mse1 = mean_squared_error(t1m, p1m)
#     #mae1 = mean_absolute_error(t1m, p1m)
#     r21 = r2_score(t1m, p1m)
#     mse2 = mean_squared_error(t2m, p2m)
#     #mae2 = mean_absolute_error(t2m, p2m)
#     r22 = r2_score(t2m, p2m)

#     exact_score_acc = np.mean((np.rint(p1m) == t1m) & (np.rint(p2m) == t2m))

#     pred_outcome = np.where(np.isclose(p1m, p2m, atol=0.2), 1, np.where(p1m > p2m, 3, 0))
#     true_outcome = preds.loc[mask, 'team_1_result'].values
#     outcome_acc = np.mean(pred_outcome == true_outcome)
    

#     eval_rows.append({
#         'model': name,
#         'team1_mse': mse1, 'team1_r2': r21,
#         'team2_mse': mse2, 'team2_r2': r22,
#         'exact_score_acc': exact_score_acc,
#         'outcome_acc': outcome_acc,
#         'n_matches': int(mask.sum())
#     })

# eval_df = pd.DataFrame(eval_rows)
# print('\nUpcoming matches evaluation:')
# print(eval_df)












# # Klasyfikatory
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report

# # Nazwy i modele klasyfikacyjne (analogicznie do regresorów)
# clf_names = [
#     "Random Forest",
#     "Decision Tree",
#     "Support Vector Classifier",
#     "Neural Network",
# ]

# classifiers = [
#     RandomForestClassifier(max_depth=5, n_estimators=100, max_features='sqrt', random_state=42),
#     DecisionTreeClassifier(max_depth=6, random_state=42),
#     make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
#     #XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='mlogloss', random_state=42),
#     make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
# ]

# # ścieżki do zapisu
# clf_save_paths = {name: f'{name}_clf.pkl' for name in clf_names}

# # etykiety do klasyfikacji
# Y_train_outcome = train_data['team_1_result']
# Y_test_outcome = test_data['team_1_result']

# # Train, save, eval
# for name, clf in zip(clf_names, classifiers):
#     clf.fit(X_train, Y_train_outcome)
#     with open(clf_save_paths[name] + '_team1', 'wb') as f:
#         pickle.dump(clf, f)

#     # predykcje i metryki
#     test_data[name + '_pred_result'] = clf.predict(X_test)
#     train_pred = clf.predict(X_train)
#     test_pred = clf.predict(X_test)

#     train_acc = accuracy_score(Y_train_outcome, train_pred)
#     test_acc = accuracy_score(Y_test_outcome, test_pred)

#     print(f'{name} Team1 outcome train_acc: {train_acc:.4f} | test_acc: {test_acc:.4f}')
#     print(f'Classification report ({name}) on test:')
#     print(classification_report(Y_test_outcome, test_pred, zero_division=0))

# for name in clf_names:
#     with open(clf_save_paths[name] + '_team1', 'rb') as f:
#         loaded_clf = pickle.load(f)
#     upcoming_matches[name + '_pred_result'] = loaded_clf.predict(upcoming_matches[features])

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# preds = upcoming_matches.copy()

# truth_cols = ['match_id','team_1_goalsScored','team_2_goalsScored','team_1_result']
# truth = match_stats[truth_cols].drop_duplicates(subset='match_id')
# preds = preds.merge(truth, on='match_id', how='left', suffixes=('','_true'))

# eval_rows = []
# for name in clf_names:
#     col = f'{name}_pred_result'   
#     if col not in preds.columns:
#         print(f'Brak kolumny predykcji dla modelu {name}: {col}')
#         continue

#     y_pred = preds[col].values
#     y_true = preds['team_1_result'].values

#     mask = ~pd.isna(y_true)
#     if mask.sum() == 0:
#         print(f'Brak prawdziwych wyników dla modelu {name}')
#         continue

#     y_pred_m = y_pred[mask]
#     y_true_m = y_true[mask]

#     acc = accuracy_score(y_true_m, y_pred_m)
#     cm = confusion_matrix(y_true_m, y_pred_m)  
#     report = classification_report(y_true_m, y_pred_m, zero_division=0)

#     print(f'\nModel: {name}')
#     print(f'Accuracy: {acc:.4f} | n_matches: {int(mask.sum())}')
#     print(report)

#     eval_rows.append({
#         'model': name,
#         'accuracy': acc,
#         'n_matches': int(mask.sum())
#     })

# eval_df = pd.DataFrame(eval_rows)
# print('\nUpcoming matches classification summary:')
# print(eval_df)






