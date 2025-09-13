# NBA Data Dashboard

This repository contains the codebase for an NBA Data Dashboard, a tool designed to visualize and analyze NBA statistics. The dashboard provides insights into player and team performance using interactive charts and tables.

## Features

- **Player Statistics**: View detailed stats for individual players.
- **Team Comparisons**: Compare team performance across various metrics.
- **Interactive Visualizations**: Explore data through dynamic charts and graphs.
- **Search and Filter**: Quickly find specific players, teams, or stats.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/nba-data-dashboard.git
    ```
2. Navigate to the project directory:
    ```bash
    cd nba-data-dashboard
    ```
3. Install dependencies:
    python -m venv venv
    source venv/bin/activate  
    pip install -r requirements.txt

4. Run ETL:
    python etl/nba_etl.py

5. Start the app:
    streamlit run app.py

## Technologies Used

- **Frontend**: streamlit
- **Backend**: Python
- **Data Source**: nba_api
- **Data Wrangling**: Pandas, NumPy
- **Similarity Search**: scikit-learn (StandardScaler + cosine similarity)
- **Visualizations**: Plotly

