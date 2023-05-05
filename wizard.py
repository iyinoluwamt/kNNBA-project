import datetime
import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor

import pandas as pd
from nba_api.stats.endpoints import playergamelog, PlayerGameLogs
from nba_api.stats.static import players as play

from modelcontainer import ModelContainer, ModelGroup
from modelcore import ModelCore


class Wizard:
    def __init__(self, raw_data_path, k_best, model_config, all_cores=True, junk=False):
        self.current_season_data = pd.read_csv("/Users/iyinoluwatugbobo/PycharmProjects/kNNBA-project/backend/data/nba_api_data.csv")
        self.k_best = k_best
        self.junk = junk
        self.core = ModelCore(raw_data_path,
                              k_best=k_best,
                              junk=junk,
                              all_cores=all_cores)
        # self.core.display_evaluations()
        self.model_groups = self._get_models(k_best, model_config)
        self.players = {}
        self.player_team_map = self._get_player_team_map()

        now = datetime.datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        self.start_time = formatted_date_time

    def _get_models(self, k_best, model_configs):
        if model_configs != {}:
            self.model_configs = model_configs
        else:
            self.model_configs = {
                'svm_rgb': ('svm_rgb', {}),
                'svm_lr': ('svm_lr', {}),
                'knn': ('knn', {},),
                'rf': ('rf', {}),
                'lr': ('lr', {}),
                'rf_500_10': ('rf', {'model__n_estimators': [500], 'model__max_depth': [10]}),
                'rf_500_20': ('rf', {'model__n_estimators': [500], 'model__max_depth': [20]}),
                'rf_500_30': ('rf', {'model__n_estimators': [500], 'model__max_depth': [30]}),
                'rf_1000_10': ('rf', {'model__n_estimators': [1000], 'model__max_depth': [10]}),
                'rf_1000_20': ('rf', {'model__n_estimators': [1000], 'model__max_depth': [20]}),
                'rf_1000_30': ('rf', {'model__n_estimators': [1000], 'model__max_depth': [30]}),
                'knn_5': ('knn', {'model__n_neighbors': [5]}),
                'knn_10': ('knn', {'model__n_neighbors': [10]}),
                'knn_15': ('knn', {'model__n_neighbors': [15]}),
                'knn_20': ('knn', {'model__n_neighbors': [20]}),
                'knn_30': ('knn', {'model__n_neighbors': [30]}),
                'knn_50': ('knn', {'model__n_neighbors': [50]}),
            }

        return {
            model_name: self._load_or_train(model_name, k_best)
            for model_name, (model_type, param_grid) in self.model_configs.items()
        }

    def _load_or_train(self, model_name, with_threads=False):
        model_type, param_grid = self.model_configs[model_name]
        file_name = f"backend/models/{model_name}.pkl"

        div = "*"
        print(f"\n\033[36m{div * 20}      {model_type}        {div * 20}\033[0m")
        if param_grid != {}:
            print(f"Params: {param_grid}\n")
        if not os.path.exists(file_name):
            return self.core.fit(
                model_name=model_name,
                model_type=model_type,
                param_grid=param_grid
            )
        else:
            return ModelGroup.load(file_name)

    @staticmethod
    def _load_all_models(directory_path):
        all_models = {}
        for filename in os.listdir(directory_path):
            if filename.endswith(".pkl"):
                model_type = filename[:-4]
                with open(os.path.join(directory_path, filename), 'rb') as f:
                    model_data = pickle.load(f)
                all_models[model_type] = model_data
        return all_models


    def _get_player_team_map(self):
        all_players = play.get_active_players()
        player_team_map = defaultdict(set)

        team_player_data = pd.read_csv('backend/data/nba_api_data.csv').dropna(subset=['PLAYER_NAME'])

        team_player_data = team_player_data.sort_values(by=['GAME_DATE'], ascending=False)
        team_player_data = team_player_data.drop_duplicates(subset=['PLAYER_NAME'], keep='first')

        for i, row in team_player_data.iterrows():
            player_team_map[row["TEAM_ABBREVIATION"]].add((row["PLAYER_NAME"], row["PLAYER_ID"]))
            self.players[row["PLAYER_NAME"]] = row["PLAYER_ID"]

        player_team_map = {team: list(players) for team, players in player_team_map.items()}
        return player_team_map

    # def _get_player_games(self, player_name, n_games=5):
    #     game_log = self.current_season_data[self.current_season_data["PLAYER_NAME"] == player_name].head(
    #         n_games)
    #
    #     if not game_log.empty:
    #         return game_log
    #     else:
    #         print(f"Player '{player_name}' not found")

    def _get_player_games(self, name, n_recent_games):
        game_log = PlayerGameLogs(
            season_nullable="2022-23",
            player_id_nullable=self.players[name],
            last_n_games_nullable=n_recent_games
        )
        time.sleep(0.5)
        game_log_df = game_log.get_data_frames()[0]
        game_log_df = game_log_df.head(n_recent_games)
        return game_log_df.dropna()

    def _calculate_discrepancy(self, player_name, model_name, n_games=3, player_stats=None):
        if player_stats is None:
            player_stats = self.project_player(player_name, model_name, n_games)

        actual_points = sum([
            player_stats["FTM"] + 3 * player_stats["FG3M"] + 2 * (player_stats["FGM"] - player_stats["FG3M"])
        ])
        return actual_points

    def project_player(self, name, model_name, n_games=3):
        in_features = self.core.in_features
        sample = self._get_player_games(name, n_games)[in_features].mean().to_frame().T

        group = self.model_groups[model_name]
        predictions = {
            "data": {},
            "name": name,
            "model_name": model_name,
            "n_games": n_games
        }
        for feature in group.models:
            predictions["data"][feature] = group.get_predictions(sample, feature)
        if not self.junk:
            ftm, fg3m, fgm = predictions["data"]["FTM"], predictions["data"]["FG3M"], predictions["data"]["FGM"]
            predictions["data"]["PTS"] = sum(
                [predictions["data"]["FTM"] +
                 (3 * predictions["data"]["FG3M"]) +
                 (2 * (predictions["data"]["FGM"] -
                       predictions["data"]["FG3M"]))]
            )
            return predictions
        else:
            return predictions

    def project_team_players(self, team_abbrev, model_name, n_games=3):
        players = [player[0] for player in self.player_team_map[team_abbrev]]
        predictions = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.project_player, player, model_name, n_games): player for player in players}

            for future in as_completed(futures):
                player = futures[future]
                try:
                    prediction = future.result()
                    predictions[player] = prediction
                except Exception as e:
                    print(f"Error while processing {player}: {e}")

        return predictions


if __name__ == "__main__":
    wizard = Wizard(
        raw_data_path="/Users/iyinoluwatugbobo/PycharmProjects/kNNBA-project/backend/data/nba.csv",
        junk=True,
        model_config={}
    )
