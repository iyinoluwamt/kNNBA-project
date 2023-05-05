import pdb
import time
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from tqdm import tqdm


seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22']
all_players = players.get_active_players()

all_game_logs = []

for player in tqdm(all_players, desc="Fetching game logs", colour="CYAN"):
    name, pid = player["full_name"], player["id"]
    for season in seasons:
        logs = playergamelog.PlayerGameLog(player_id=pid, season=season, season_type_all_star="Regular Season")
        try:
            df = logs.get_data_frames()[0]
            all_game_logs.append(df)
        except Exception as e:
            print(f"Error fetching game logs for player {name}: {e}")
        time.sleep(1)  # add a delay between requests to reduce load on the API

# Concatenate all game logs into a single DataFrame
all_game_logs_df = pd.concat(all_game_logs, ignore_index=True)

# Export the DataFrame to a CSV file
all_game_logs_df.to_csv('nba_training.csv', index=False)
