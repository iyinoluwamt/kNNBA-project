import math
import numbers
import os
import pickle

from sklearn.pipeline import Pipeline
from tabulate import tabulate


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
        return math.ceil(y_pred[0])

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
