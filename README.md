# College Football Game Predictor

A machine learning-powered tool that predicts the outcome and expected score of college football games using historical data and team statistics. The predictor uses a Random Forest Classifier to determine the winner between two teams and provides an estimated point differential based on key features such as offensive and defensive performance, turnover margin, and home-field advantage.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Example](#example)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features
- Predict the winner between two college football teams.
- Predict the projected points for both teams, including a home-field advantage feature.
- User-friendly graphical interface built with Tkinter for ease of use.

## Technologies Used
- **Python**: Core programming language.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning model (Random Forest Classifier).
- **Tkinter**: For building the graphical user interface (GUI).

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/college-football-predictor.git
    cd college-football-predictor
    ```

2. **Install required packages**:
    Install the necessary Python libraries via `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have Python installed (Python 3.7 or higher recommended).*

3. **Download the dataset**:
   The predictor uses historical data from various CSV files. Ensure the required CSV files are present in the `archive/` directory.

## Usage

1. **Run the GUI**:
    ```bash
    python cfb_gui.py
    ```

2. **Input Teams**:
    - Enter the home team and away team names in the GUI.
    - Click the "Predict Winner" button.

3. **View Results**:
    The GUI will display the predicted winner, along with an estimated point differential for the game. 

## Data

The prediction is based on various features such as:
- Offensive yards per game
- Defensive yards allowed
- Points per game
- Turnover margin
- Third-down conversion rates
- Home-field advantage, with the option to input home and away teams.

## Example

To predict a game between **Alabama (SEC)** and **Tennessee (SEC)**:

1. Enter **Alabama (SEC)** as the home team.
2. Enter **Tennessee (SEC)** as the away team.
3. Click **Predict Winner**.
4. View the result: "Alabama wins by 7 points".

## Project Structure

```bash
.
├── archive/                     # Folder containing historical CSV data for each year
├── cfb_gui.py                   # Main Python file to launch the GUI
├── cfb_model.py                 # Core machine learning logic and predictions
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
