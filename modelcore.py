import math
import numbers
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from tabulate import tabulate
from tqdm import tqdm


class ModelCore:
    def __init__(self, raw_data_path, k_best=5, junk=False, all_cores=False):
        if junk:
            k_best = 2
            all_cores = False

        self.raw_data = pd.read_csv(raw_data_path)
        self.junk = junk
        self.k_best = k_best
        self.all_cores = all_cores

        self.out_features, self.in_features = self.get_features()
        self.scalers = {}
        self.data = None

    def get_features(self):
        if self.junk:
            out_features = ["PTS", "REB", "AST", "MIN", "TOV"]
            in_features = out_features
        else:
            out_features = ["OREB", "DREB", "AST", "FG3A", "FGA", "FTA",
                            "FG3M", "FGM", "FTM", "STL", "BLK", "MIN", "PF"]
            in_features = out_features + ["PTS", "REB", "FG3_PCT", "FG_PCT", "FT_PCT", "TOV", "PLUS_MINUS"]
        return out_features, in_features

    def process_raw_data(self):
        self.raw_data['H'] = 0
        self.raw_data['A'] = 0
        self.raw_data.loc[self.raw_data['MATCHUP'].str.contains(' vs. '), 'H'] = 1
        self.raw_data.loc[self.raw_data['MATCHUP'].str.contains(' @ '), 'A'] = 1
        self.raw_data['W'] = self.raw_data['WL'].apply(lambda x: 1 if x == 'W' else 0)
        self.raw_data.drop('WL', axis=1, inplace=True)

    def get_training_data(self, test_size):
        if self.data is None:
            self.process_raw_data()
            self.data = self.raw_data[self.in_features]
            self.data = self.data.dropna().copy()

        training_data = {}
        for feature in self.out_features:
            X, y = self.data.loc[:, self.data.columns != feature], self.data[feature]
            training_data[feature] = train_test_split(X, y, test_size=test_size)
        return training_data

    @staticmethod
    def _export(name, model_type, feature, model, eval):
        return {
            'name': name,
            'type': model_type,
            'feature': feature,
            'model': model,
            'eval': eval,
            "params": model.get_params()
        }

    @staticmethod
    def export_model_group(group, model_name):
        model_dir = "backend/models"
        os.makedirs(model_dir, exist_ok=True)

        file_name = f"{model_dir}/{model_name}.pkl"
        group.save(file_name)

    def export_scalers(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(self.scalers, f)

    def fit(self, model_name, model_type='lr', param_grid={}):
        model_mapping = {
            'lr': LinearRegression(),
            'rf': RandomForestRegressor(),
            'knn': KNeighborsRegressor(),
            'svm_lr': SVR(kernel='linear'),
            'svm_poly': SVR(kernel='poly'),
            'svm_rbf': SVR(kernel='rbf')
        }
        model = model_mapping.get(model_type, LinearRegression())

        all_models = {}
        pbar = tqdm(self.out_features, desc=f"Fitting {model_name}", colour="CYAN")
        for feature in pbar:
            X_train, X_test, y_train, y_test = self.get_training_data(test_size=0.2)[feature]

            pipeline = Pipeline([
                ('k_best', SelectKBest(k=self.k_best)),
                ('model', model)
            ])

            n_jobs = -1 if self.all_cores else 1
            grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=n_jobs)
            grid_search.fit(X_train, y_train)

            eval = self._evaluate(feature, grid_search, X_train, X_test, y_test)
            model_data = self._export(model_name, model_type, feature, grid_search.best_estimator_, eval)
            all_models[f"{feature}"] = model_data

        model_group = ModelGroup(
            name=model_name,
            models={key: model_data["model"] for key, model_data in all_models.items()},
            evaluations={key: model_data["eval"] for key, model_data in all_models.items()}
        )
        self.export_model_group(model_group, model_name)
        model_group.display_evaluations_table()

    @staticmethod
    def _evaluate(feature, model, X_train, X_test, y_test):
        evaluation = {}

        # Get selected features
        selected_features = model.best_estimator_.named_steps['k_best'].get_support()
        X_train_selected = X_train.loc[:, selected_features]
        X_test_selected = X_test.loc[:, selected_features]

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        evaluation['MSE'] = mse
        evaluation['R2'] = r2
        evaluation['MAE'] = mae
        evaluation['selected_features'] = X_train_selected.columns.tolist()
        evaluation['params'] = model.best_params_
        evaluation['best_score'] = model.best_score_

        return evaluation

    def display_evaluations(self):
        model_dir = "backend/models"

        if not os.path.exists(model_dir):
            print("No models directory found.")
            return

        for filename in os.listdir(model_dir):
            if filename.endswith(".pkl"):
                model_name = filename[:-4]
                model_group = ModelGroup.load(self, os.path.join(model_dir, filename))
                print(f"Evaluations for {model_name}:")
                for feature, evaluation in model_group.evaluations.items():
                    print(f"  {feature}:")
                    for metric, value in evaluation.items():
                        print(f"    {metric}: {value:.4f}")
                print()


class ModelContainer:
    def __init__(self, model_type, models, evaluations, params):
        self.model_type = model_type
        self.models = models
        self.evaluations = evaluations
        self.params = params

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return cls(
            model_type=data["model_type"],
            models=data["models"],
            evaluations=data["evaluations"],
            params=data["params"]
        )

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump({
                "model_type": self.model_type,
                "models": self.models,
                "evaluations": self.evaluations,
                "params": self.params
            }, f)

    @staticmethod
    def load_all_models(directory_path):
        all_models = {}
        for filename in os.listdir(directory_path):
            if filename.endswith(".pkl"):
                model_type = filename[:-4]
                model_data = ModelContainer.load(os.path.join(directory_path, filename))
                all_models[model_type] = model_data
        return all_models


class ModelGroup:
    def __init__(self, name, models, evaluations):
        self.name = name
        self.models = models
        self.evaluations = evaluations

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_predictions(self, X, feature):
        estimator = self.models[feature]
        predictions = {}
        X_without_feature = X.drop(columns=[feature])
        pipeline = Pipeline([
            ('k_best', estimator.named_steps['k_best']),
            ('model', estimator.named_steps['model'])
        ])
        y_pred = pipeline.predict(X_without_feature)
        return math.floor(y_pred[0])

    def display_evaluations_table(self):
        headers = ["Feature"] + list(next(iter(self.evaluations.values())).keys())
        table_data = []

        for feature, evaluation in self.evaluations.items():
            row = [feature] + [round(value, 4) if isinstance(value, numbers.Number) else value
                               for value in evaluation.values()]
            table_data.append(row)

        table = tabulate(table_data, headers=headers, tablefmt="grid")
        print(f"\nEvaluations for {self.name}:")
        print(table)
