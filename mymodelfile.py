from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from ast import literal_eval
import numpy as np
import pandas as pd


class MyModel:
    def __init__(self):
        pass

        # print("Yes")

    def fit(self, lst):

        ipl_matches_result = pd.read_csv(
            "IPL_Matches_Result_2008_2022.csv")
        ipl_ball_by_ball = pd.read_csv(
            "IPL_Ball_by_Ball_2008_2022.csv")
        ipl_result2 = ipl_matches_result.drop(
            columns=['City', 'Date', 'SuperOver', 'MatchNumber', 'Umpire1', 'Umpire2', 'method'])
        ipl_ball2 = ipl_ball_by_ball.drop(
            columns=['kind', 'fielders_involved', 'non_boundary'])
        runs_by_batsman = ipl_ball2.groupby('batter')['batsman_run'].sum()

        # calculate number of matches played by each batsman
        matches_by_batsman = ipl_ball2.groupby('batter')['ID'].nunique()

        # calculate average runs by each batsman
        average_runs_by_batsman = runs_by_batsman / matches_by_batsman

        # create a dataframe with batsman names, total runs, number of matches played and average runs
        batsman_stats = pd.DataFrame({
            'Batsman': runs_by_batsman.index,
            'Total Runs': runs_by_batsman.values,
            'Matches Played': matches_by_batsman.values,
            'Average Runs': average_runs_by_batsman.values
        })
        # get a dictionary with batsman names as keys and their average runs as values
        batsman_avg = dict(
            zip(batsman_stats['Batsman'], batsman_stats['Average Runs']))

        # replace batsman names in ipl_ball_by_ball with their average runs
        ipl_ball2['batter'] = ipl_ball2['batter'].apply(
            lambda x: batsman_avg.get(x, x))
        ipl_ball2['non-striker'] = ipl_ball2['non-striker'].apply(
            lambda x: batsman_avg.get(x, x))

        ipl_ball2 = ipl_ball2.drop(columns='player_out')
        ipl_ball2 = ipl_ball2.drop(columns='extra_type')

        # ipl_ball2['bowler'].groupby(ipl_ball2['bowler'].lam)
        wicket_by_bowlers = ipl_ball2.groupby(
            'bowler')['isWicketDelivery'].sum()
        wicket_by_bowlers
        matches_by_bowlers = ipl_ball2.groupby('bowler')['ID'].nunique()
        matches_by_bowlers
        avg_wickets_over_matches_by_bowlers = wicket_by_bowlers/matches_by_bowlers
        avg_wickets_over_matches_by_bowlers
        # ipl_ball2['bowler'].apply(lambda x :x for x in avg_wickets_over_matches_by_bowlers  )
        ipl_ball2['bowler'] = ipl_ball2['bowler'].map(
            avg_wickets_over_matches_by_bowlers)

        ipl_result2['TossDecision'].replace(
            {'bat': 1, 'field': 0}, inplace=True)

        # Filter the ipl_ball_by_ball dataset to only include the overs up to 6
        ipl_ball_6overs = ipl_ball_by_ball.loc[ipl_ball_by_ball['overs'] <= 6]

        # Group the filtered dataset by match_id and batting_team and sum the runs
        runs_6overs = ipl_ball_6overs.groupby(['ID', 'BattingTeam'])[
            'total_run'].sum().reset_index()
        merged_data = pd.merge(ipl_result2, runs_6overs, on=['ID'], how='left')
        merged_data['BowlingTeam'] = merged_data['BattingTeam']
        batting_team_col = merged_data['BattingTeam']
        for i in range(len(merged_data['BattingTeam'])):
            if merged_data['BattingTeam'][i] == merged_data['Team1'][i]:
                # merged_data['BowlingTeam'][i] = merged_data['Team2'][i]
                merged_data.loc[i, 'BowlingTeam'] = merged_data.loc[i, 'Team2']
            else:
                # merged_data['BowlingTeam'][i] = merged_data['Team1'][i]
                merged_data.loc[i, 'BowlingTeam'] = merged_data.loc[i, 'Team1']

        merged_data['TossDecision'].replace(
            {'bat': 1, 'field': '0'}, inplace=True)

        merged_data.replace({'Delhi Daredevils': 'Delhi Capitals',
                            'Kings XI Punjab': 'Punjab Kings',
                             'Rising Pune Supergiant': 'Rising Pune Supergiants'},
                            inplace=True)
        # now task is to replace these values with some numerical data
        team_name_index = {"Mumbai Indians": 30,
                           "Chennai Super Kings": 28,
                           "Kolkata Knight Riders": 26,
                           "Royal Challengers Bangalore": 24,
                           "Delhi Capitals": 22,
                           "Punjab Kings": 20,
                           "Rajasthan Royals": 18,
                           "Sunrisers Hyderabad": 16,
                           "Deccan Chargers": 14,
                           "Rising Pune Supergiants": 12,
                           "Gujarat Lions": 10,
                           "Gujarat Titans": 8,
                           "Pune Warriors": 6,
                           "Lucknow Super Giants": 4,
                           "Kochi Tuskers Kerala": 2}
        merged_data.replace(team_name_index, inplace=True)

        merged_data["WonBy"].replace(
            {'Wickets': 4, 'Runs': 6, 'SuperOver': 2, 'NoResults': 4}, inplace=True)

        merged_data['Player_of_Match'] = merged_data['Player_of_Match'].apply(
            lambda x: batsman_avg.get(x, 0) if x in batsman_avg else 0)

        # Step 1: Create dictionary with average runs for each venue
        venue_avg = merged_data.groupby('Venue')['total_run'].mean().to_dict()

        # Step 2: Sort the dictionary in descending order of average runs
        venue_avg = {k: v for k, v in sorted(
            venue_avg.items(), key=lambda item: item[1], reverse=True)}

        # Step 3: Create a list of venues in the sorted order
        sorted_venues = list(venue_avg.keys())

        # Step 4: Create dictionary with venue encodings
        venue_encodings = {venue: label for label,
                           venue in enumerate(sorted_venues)}

        # Step 5: Assign labels to venues
        for venue in sorted_venues:
            venue_encodings[venue] = sorted_venues.index(venue)

        # Step 6: Map venue names to their labels in the ipl_matches dataset
        merged_data['Venue'] = merged_data['Venue'].map(venue_encodings)

        merged_data['Season'] = merged_data['Season'].replace(
            {"2009/10": "2010", "2007/08": "2008", "2020/21": "2020"})
        merged_data['Season'] = merged_data['Season'].apply(lambda x: int(x))

        # iterate over each row
        for index, row in merged_data.iterrows():
            # use ast.literal_eval to safely evaluate the string as a list
            players_list = literal_eval(row['Team1Players'])
            players_list2 = literal_eval(row['Team2Players'])
            # sort the players by their average scores
            sorted_players = sorted(
                players_list, key=lambda x: batsman_avg.get(x, 0), reverse=True)
            sorted_players2 = sorted(
                players_list2, key=lambda x: batsman_avg.get(x, 0), reverse=True)
            # take only the top 4 players
            top_players = sorted_players[:4]
            top_players2 = sorted_players2[:4]

            # update the value in the dataframe with the new list of players
            merged_data.at[index, 'Team1Players'] = top_players
            merged_data.at[index, 'Team2Players'] = top_players2

        team1_players = pd.get_dummies(pd.DataFrame(
            merged_data['Team1Players'].tolist()).stack()).groupby(level=0).sum()
        team2_players = pd.get_dummies(pd.DataFrame(
            merged_data['Team2Players'].tolist()).stack()).groupby(level=0).sum()

        players_name = list(team1_players.columns)+list(team2_players.columns)
        merged_data = merged_data.drop(
            columns=["Team1Players", "Team2Players"])
        # print(len(players_name))

        mapping = dict(zip(batting_team_col, merged_data['BattingTeam']))

        merged_data['WinningTeam'] = merged_data['WinningTeam'].fillna(0)
        merged_data['Margin'] = merged_data['Margin'].fillna(0)

        merged_data_encoded = pd.concat(
            [merged_data, team1_players, team2_players], axis=1)

        # reading input test file and applying some conventions on it
        test_file = pd.read_csv('test_file.csv')

        merged_data_encoded = merged_data_encoded.drop(
            columns=['ID', 'Season', 'Team1', 'Team2', 'TossWinner', 'TossDecision', 'WinningTeam', 'WonBy', 'Margin', 'Player_of_Match'])
        training_file = merged_data_encoded
        # to add inning column to datagrame
        innings_list = [1, 2] * (len(training_file) // 2) + \
            [1] * (len(training_file) % 2)
        training_file['innings'] = innings_list

        test_file.rename(columns={"venue": "Venue", 'batting_team': 'BattingTeam',
                         'bowling_team': 'BowlingTeam'}, inplace=True)
        test_file['Venue'] = test_file['Venue'].map(venue_encodings)
        mapping = dict(zip(batting_team_col, test_file['BattingTeam']))
        zero_data = np.zeros(shape=(len(test_file), len(players_name)))
        zero_data = zero_data.astype(int)

        d = pd.DataFrame(zero_data, columns=players_name)
        # print(d)
        row1 = test_file['batsmen'][0].split(", ")
        row2 = test_file['batsmen'][1].split(", ")
        t_col = training_file.columns
        for i in range(0, len(d.columns)):
            if t_col[i] in row1:
                d[0][i] = 1
                print(d[0][i])

        t_col = training_file.columns
        for i in range(0, len(d.columns)):
            if t_col[i] in row2:
                d[1][i] = 1
                print(d[0][i])

        # print(d.sum())
        test_file = test_file.join(d)

        # <----------------------->
        # # Create a list of all unique batsmen and bowlers
        test_file = test_file.drop(columns=['bowlers'])
        test_file.replace(team_name_index, inplace=True)
        test_file = test_file.drop(columns=['batsmen'])
        inninigs = test_file['innings']
        test_file = test_file.drop(columns=['innings'])
        test_file['innings'] = inninigs

        runs = training_file['total_run']
        training_file = training_file.drop(columns=['total_run'])
        training_file['total_run'] = runs

        X = training_file.drop('total_run', axis=1)
        y = training_file['total_run']
        training_file = X

        data = [training_file, test_file]
        training_file = pd.concat(data, ignore_index=True, sort=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        # self.fit(X_train, y_train)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        self.predict(X_test)

        # print("Fit Working")
        # Train the parameters of the model using the training data
        # X_train is the input training data as a pandas dataframe
        # y_train is the target training data as a pandas dataframe
        # Return a reference to the MyModel object itself
        return self

    def predict(self, X_test):
        # Make predictions using the trained model
        # X_test is the input test data as a pandas dataframe
        # Return the predictions in the specified format
        y_pred = self.model.predict(X_test)

        last_two_preds = y_pred[-2:]
        for i in range(len(last_two_preds)):
            last_two_preds[i] = int(last_two_preds[i])
        # create a pandas dataframe with id and predicted_runs columns
        df = pd.DataFrame({'id': [0, 1], 'predicted_runs': last_two_preds})
        # save the dataframe to a csv file
        df.to_csv('submission.csv', index=False)
        # print("pred")
        return y_pred[-2:]


model = MyModel()
