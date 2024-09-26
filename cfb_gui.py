import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox

# Load data
file_data = {
    'archive/cfb18.csv': 2018,
    'archive/cfb19.csv': 2019,
    'archive/cfb20.csv': 2020,
    'archive/cfb21.csv': 2021,
    'archive/cfb22.csv': 2022,
    'archive/cfb23.csv': 2023
}

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

# Apply recent weights
def apply_recent_weights(df):
    weighted_data = pd.concat([
        df['Win'] * df['Year'].map(recent_weights),
        df['Loss'] * df['Year'].map(recent_weights),
        df['Points.Per.Game'] * df['Year'].map(recent_weights)
    ], axis=1)
    
    weighted_data.columns = ['Weighted.Win', 'Weighted.Loss', 'Weighted.Points.Per.Game']
    return pd.concat([df, weighted_data], axis=1)

# Apply the recent performance weights to the dataset
data = apply_recent_weights(data)

# Features to use for prediction
features = [
    'Weighted.Win', 'Weighted.Loss', 'Off.Yards.per.Game', 'Yards.Allowed',
    'Turnover.Margin', 'X3rd.Percent', 'Opponent.3rd.Percent'
]

# Standardize numerical features
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Target variable is points per game
target = 'Points.Per.Game'

# Train-test split to fit a regression model
X = data[features]
y = data[target]  # Use Points Per Game as the target for regression

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor to predict points scored
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Homefield advantage multiplier (you can adjust this value)
homefield_advantage = 1.05  # 5% boost for the home team

# Function to predict the number of points a team will score based on the trained model
def calculate_points(team):
    team_data = data[data['Team'] == team]
    
    if team_data.empty:
        return None  # Return None if team not found
    
    team_features = team_data[features].mean()
    return model.predict([team_features])[0]  # Return the predicted points

# Function to predict the winner between a home team and an away team
def predict_winner(home_team, away_team):
    points_home = calculate_points(home_team)
    points_away = calculate_points(away_team)
    
    if points_home is None or points_away is None:
        return "One or both team names are incorrect!"
    
    # Apply homefield advantage to the home team's predicted points
    points_home *= homefield_advantage
    
    # Compare points to predict winner
    if points_home > points_away:
        return f"{home_team} wins with {points_home:.2f} points vs. {away_team} with {points_away:.2f} points!"
    elif points_away > points_home:
        return f"{away_team} wins with {points_away:.2f} points vs. {home_team} with {points_home:.2f} points!"
    else:
        return f"It's a draw! {home_team}: {points_home:.2f} points, {away_team}: {points_away:.2f} points."

# Tkinter GUI setup
def run_gui():
    def get_prediction():
        home_team = home_team_entry.get()
        away_team = away_team_entry.get()
        if not home_team or not away_team:
            messagebox.showerror("Input Error", "Please enter both team names!")
            return
        result = predict_winner(home_team, away_team)
        result_label.config(text=result)
    
    # Main window
    window = tk.Tk()
    window.geometry("500x500")
    window.title("College Football Game Predictor Model")
    
    # Home team input
    home_team_label = tk.Label(window, text="Enter Home Team:", font=('Arial', 18))
    home_team_label.pack()
    home_team_entry = tk.Entry(window)
    home_team_entry.pack()

    # Away team input
    away_team_label = tk.Label(window, text="Enter Away Team:", font = ('Arial', 18))
    away_team_label.pack()
    away_team_entry = tk.Entry(window)
    away_team_entry.pack()

    # Predict button
    predict_button = tk.Button(window, text="Predict Winner", font=('Arial', 18), command=get_prediction, bg = "blue", fg = "white")
    predict_button.pack(pady = (10, 10))

    # Result label
    result_label = tk.Label(window, text="")
    result_label.pack(pady = (5, 5))

    # Run the window loop
    window.mainloop()

# Run the GUI
if __name__ == "__main__":
    run_gui()
