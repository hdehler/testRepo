# USAGE:
# python3 cfbOriginal.py --team1 "Arizona St. (Pac-12)" --team2 "Utah (Pac-12)"  
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
    'Win-Loss': 2, 
    'Off Yards per Game': 1, 
    'Yards Per Game Allowed': -1, 
    '3rd Percent': 1, 
    'Opponent 3rd Percent': -1, 
    'Points Per Game': 2, 
    'Avg Points per Game Allowed': -1, 
    'Turnover Margin': 1
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

def calculate_score(team):
    # Check if the exact match for the team name is present in the data
    if team in data['Team'].unique():
        team_full_name = team
    else:
        # Fall back to the original logic if the exact match is not found
        matching_teams = [t for t in data['Team'].unique() if team in t.split(' ')[0]]
        if not matching_teams:
            raise ValueError(f"No matching team found for {team}.")
        team_full_name = matching_teams[0]
    
    score = 0
    team_data = data[data['Team'] == team_full_name].select_dtypes(include=[np.number]).mean()
    for metric, weight in weights.items():
        score += team_data.get(metric, 0) * weight  # Use get method to avoid KeyError
    return score


def predict_winner(team1, team2):
    score1 = calculate_score(team1)
    score2 = calculate_score(team2)
    
    # Debugging outputs:
    print(f"Score for {team1}: {score1}")
    print(f"Score for {team2}: {score2}")
    
    # Commenting out the spread calculation
    # scaled_difference = abs(score1 - score2) / 10  
    # point_spread = round(scaled_difference, 1)
    
    if score1 > score2:
        return team1
    elif score2 > score1:
        return team2
    else:
        return "Draw"

def main():
    parser = argparse.ArgumentParser(description="Predict the winner between two college football teams.")
    parser.add_argument("--team1", required=True, help="First team to compare.")
    parser.add_argument("--team2", required=True, help="Second team to compare.")
    args = parser.parse_args()

    # Debugging: print unique teams in dataset to ensure team names are present and correctly spelled
    teams_in_data = data['Team'].unique()

    print(predict_winner(args.team1, args.team2))

if __name__ == "__main__":
    main()
