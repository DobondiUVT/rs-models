import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from openbox import Optimizer
from src.utils.constants import GLOBAL_RANDOM_STATE, OPENBOX_SPACE, MODEL_REGISTRY


class OpenboxModel:
    def __init__(self, model_type, target='HeartDisease', max_runs=50):
        self.model_type = model_type
        self.target = target
        self.max_runs = max_runs
        self.best_model = None
        self.best_params = None

    def _build_pipeline(self, clf):
        return Pipeline([
            ('preprocessor', ColumnTransformer([
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), lambda df: df.select_dtypes(include=['int64', 'float64']).columns),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), lambda df: df.select_dtypes(include=['object']).columns)
            ])),
            ('classifier', clf)
        ])

    def train(self, data_path):
        df = pd.read_csv(data_path)
        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=GLOBAL_RANDOM_STATE
        )

        def objective(config):
            params = dict(config)
            clf = MODEL_REGISTRY[self.model_type](**params)
            pipe = self._build_pipeline(clf)
            score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
            return {'objectives': [-score]}

        space = OPENBOX_SPACE[self.model_type]
        optimizer = Optimizer(
            objective,
            space,
            max_runs=self.max_runs,
            random_state=GLOBAL_RANDOM_STATE,
            task_id=f'{self.model_type}_optimization'
        )

        history = optimizer.run()

        # Get best configuration from history
        best_observation = history.get_incumbents()[0]
        self.best_params = dict(best_observation.config)
        clf = MODEL_REGISTRY[self.model_type](**self.best_params)
        self.best_model = self._build_pipeline(clf)
        self.best_model.fit(X_train, y_train)

        test_score = self.best_model.score(X_test, y_test)
        print(f"{self.model_type} optimized accuracy: {test_score:.4f}")
        return test_score


def evaluate_models(dataset_path, target):
    results = []

    for name in MODEL_REGISTRY:
        model = OpenboxModel(name, target)
        score = model.train(dataset_path)
        results.append((name, score))
        print(f"{name:15} Accuracy: {score:.4f}")

    return results


def save_results_to_csv(results, filename='model_performance_openbox.csv'):
    df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    results = evaluate_models('../datasets/heart.csv', target='HeartDisease')
    save_results_to_csv(results)