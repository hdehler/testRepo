# USAGE:
# python3 cfbOriginal.py --team1 "Arizona St. (Pac-12)" --team2 "Utah (Pac-12)"
import pandas as pd
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# List all your CSV files and their corresponding years
file_data = {
    'archive/cfb18.csv': 2018,
    'archive/cfb19.csv': 2019,
    'archive/cfb20.csv': 2020,
    'archive/cfb21.csv': 2021,
    'archive/cfb22.csv': 2022,
    'archive/cfb23.csv': 2023
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

# Define recent weights for each year (heavier weight for more recent years)
recent_weights = {
    2018: 0.5,
    2019: 0.7,
    2020: 0.9,
    2021: 1.1,
    2022: 1.3,
    2023: 1.5
}

# Function to apply recent weights to the dataset
def apply_recent_weights(df):
    # Use pd.concat to avoid fragmentation warnings
    weighted_data = pd.concat([
        df['Win'] * df['Year'].map(recent_weights),
        df['Loss'] * df['Year'].map(recent_weights),
        df['Points.Per.Game'] * df['Year'].map(recent_weights)
    ], axis=1)
    
    weighted_data.columns = ['Weighted.Win', 'Weighted.Loss', 'Weighted.Points.Per.Game']
    return pd.concat([df, weighted_data], axis=1)

# Apply the recent performance weights to the dataset
data = apply_recent_weights(data)

# Correct column names based on your actual dataset
# Based on your dataset printout, use 'X3rd.Percent' and 'Opponent.3rd.Percent'
features = [
    'Weighted.Win', 'Weighted.Loss', 'Off.Yards.per.Game', 'Yards.Allowed',
    'Points.Per.Game', 'Turnover.Margin', 'X3rd.Percent', 'Opponent.3rd.Percent'
]

# Standardize numerical features
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Train-test split to fit a machine learning model
X = data[features]
y = data['Win']  # Use 'Win' as the target for binary classification

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier to improve prediction accuracy
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Print the accuracy of the model on the test data
print(f"Model Accuracy on Test Data: {model.score(X_test, y_test)}")

# Function to calculate score for a team based on the trained model
def calculate_score(team):
    team_data = data[data['Team'] == team]
    
    if team_data.empty:
        return 0
    
    team_features = team_data[features].mean()
    return model.predict_proba([team_features])[0][1]  # Return the probability of winning

# Function to predict the winner between two teams
def predict_winner(team1, team2):
    score1 = calculate_score(team1)
    score2 = calculate_score(team2)
    
    # Output scores for both teams
    print(f"Score for {team1}: {score1}")
    print(f"Score for {team2}: {score2}")
    
    # Compare scores to predict winner
    if score1 > score2:
        return team1
    elif score2 > score1:
        return team2
    else:
        return "Draw"

# Main function to handle user input and predict the game outcome
def main():
    parser = argparse.ArgumentParser(description="Predict the winner between two college football teams.")
    parser.add_argument("--team1", required=True, help="First team to compare.")
    parser.add_argument("--team2", required=True, help="Second team to compare.")
    args = parser.parse_args()

    print(predict_winner(args.team1, args.team2))

if __name__ == "__main__":
    main()
