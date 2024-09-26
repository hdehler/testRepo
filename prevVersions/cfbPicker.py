import pandas as pd
import argparse
import numpy as np

# List all your csv files and their corresponding years
file_data = {
    'archive/cfb17.csv': 2017,
    'archive/cfb18.csv': 2018,
    'archive/cfb19.csv': 2019,
    'archive/cfb20.csv': 2020,
    'archive/cfb21.csv': 2021,
    'archive/cfb22.csv': 2022
}

# Load dataframes with a Year column
dfs = []
for file, year in file_data.items():
    df = pd.read_csv(file)
    df['Year'] = year  # Add a Year column
    dfs.append(df)

# Concatenate them into one big dataframe
data = pd.concat(dfs, ignore_index=True)

# Handle potential NaN values (replace with 0 for simplicity)
data.fillna(0, inplace=True)

# Define weights for each metric.
weights = {
    'Off Rank': 2.5, 'Games': 1.1, 'Win-Loss': 2.8, 'Off Plays': 1.5, 'Off Yards': 2.5,
    'Off Yards/Play': 1.8, 'Off TDs': 2.3, 'Off Yards per Game': 2.2, 'Def Rank': 3,
    'Def Plays': 1.5, 'Yards Allowed': 2.2, 'Yards/Play Allowed': 1.7, 'Off TDs Allowed': 2.5,
    'Total TDs Allowed': 2.4, 'Yards Per Game Allowed': 2, '3rd Down Rank': 2.5,
    '3rd Attempts': 1.2, '3rd Conversions': 1.8, '3rd Percent': 2.3, '3rd Down Def Rank': 2.5,
    'Opp 3rd Conversion': 1.5, 'Opp 3rd Attempt': 1.2, 'Opponent 3rd Percent': 1.8,
    '4th Down Rank': 2, '4th Attempts': 1.4, '4th Conversions': 2, '4th Percent': 2,
    '4rd Down Def Rank': 2, 'Opp 4th Conversion': -1.5, 'Opp 4th Attempt': 1.3, 'Opponent 4th Percent': 1.7,
    'Penalty Rank': -1.2, 'Penalties': -1.5, 'Penalty Yards': -1.5, 'Penalty Yards Per Game': -1.5,
    'First Down Def Rank': 1.5, 'Opp First Down Runs': -1.2, 'Opp First Down Passes': -1.1,
    'Opp First Down Penalties': 1.2, 'Opp First Downs': 1.5, 'First Down Rank': 2, 'First Down Runs': 1.6,
    'First Down Passes': 1.4, 'First Down Penalties': 1.2, 'First Downs': 1.8, 'Kickoff Return Def Rank': 1.2,
    'Opp Kickoff Returns': 0.3, 'Kickoff Touchbacks': 0.5, 'Opponent Kickoff Return Yards': 0.5,
    'Opp Kickoff Return Touchdowns Allowed': 0.7, 'Avg Yards per Kickoff Return Alloweed': 0.4,
    'Kickoff Return Rank': 0.5, 'Passing Off Rank': 2, 'Pass Attempts': 1.2, 'Pass Completions': 1.8,
    'Interceptions Thrown_x': -2.5, 'Pass Yards': 2.5, 'Pass Yards/Attempt': 2.2, 'Yards/Completion': 1.5,
    'Pass Touchdowns': 2, 'Pass Yards Per Game': 2.3, 'Pass Def Rank': 2.5, 'Opp Completions Allowed': -1.5,
    'Opp Pass Attempts': 1.4, 'Opp Pass Yds Allowed': -2, 'Opp Pass TDs Allowed': -2.2, 'Redzone Def Rank': 2.8,
    'Opp Redzone Attempts': -1.8, 'Opp Redzone Rush TD Allowed': -2, 'Opp Redzone Pass Touchdowns Allowed': -2,
    'Opp Redzone Field Goals Made': -1.5, 'Redzone Points Allowed': -2.5, 'Redzone Off Rank': 2.8, 'Redzone Attempts': 2.2,
    'Redzone Rush TD': 2, 'Redzone Pass TD': 2.5, 'Redzone Field Goals Made': 2, 'Redzone Scores': 2.8, 'Redzone Points': 2.7,
    'Tackle for Loss Rank': 2.5, 'Solo Tackle For Loss': 1.8, 'Assist Tackle For Loss': 1.6, 'Average Sacks per Game': 2.5, 'Sack Rank': 2.8,
    'Sack Yards': 2, 'Sacks per Game': 2.7, 'Total Points': 2.5, 'Points Per Game': 3, 'Tackle for Loss Yards': 2, 'Average Time of Possession per Game': 2.8,
    'Turnover Rank': 3, 'Turnovers Lost': -3, 'Time of Possession Rank': 2.2
    # ... (You can extend this pattern for additional features if needed.)
}

# Define recent weights
recent_weights = {
    2017: 0.5,
    2018: 0.7,
    2019: 0.9,
    2020: 1.1,
    2021: 1.3,
    2022: 1.5
}

# You can adjust this based on how significant you believe the home advantage is.
HOME_ADVANTAGE_WEIGHT = 100
GAMETIME_WEIGHTS = {
    'noon': 0,
    'mid': 0.5,
    'night': 2
}


def calculate_score(team):
    if team in data['Team'].unique():
        team_full_name = team
    else:
        matching_teams = [t for t in data['Team'].unique() if team in t.split(' ')[
            0]]
        if not matching_teams:
            raise ValueError(f"No matching team found for {team}.")
        team_full_name = matching_teams[0]

    score = 0
    team_data = data[data['Team'] == team_full_name].select_dtypes(include=[
                                                                   np.number]).mean()
    for metric, weight in weights.items():
        # Use get method to avoid KeyError
        score += team_data.get(metric, 0) * weight
    return score


def predict_winner(home_team, away_team, gametime):
    home_score = calculate_score(home_team) + HOME_ADVANTAGE_WEIGHT + GAMETIME_WEIGHTS[gametime]
    away_score = calculate_score(away_team)

    # Difference between the teams' scores (not scaled)
    score_difference = home_score - away_score
    scaled_difference = score_difference / 10
    base_score = 28

    home_team_score = base_score + scaled_difference
    away_team_score = base_score - scaled_difference

    print(f"Projected Score for {home_team} (Home): {round(home_team_score)}")
    print(f"Projected Score for {away_team} (Away): {round(away_team_score)}")

    if home_score > away_score:
        return home_team
    elif away_score > home_score:
        return away_team
    else:
        return "Draw"


def main():
    parser = argparse.ArgumentParser(
        description="Predict the winner between two college football teams.")
    parser.add_argument("--home", required=True, help="Home team.")
    parser.add_argument("--away", required=True, help="Away team.")
    parser.add_argument("--gametime", choices=['noon', 'mid', 'night'],
                        required=True, help="Time of the game (noon, mid, night).")
    args = parser.parse_args()

    # Debugging: print unique teams in dataset to ensure team names are present and correctly spelled
    teams_in_data = data['Team'].unique()

    print(predict_winner(args.home, args.away, args.gametime))


if __name__ == "__main__":
    main()
