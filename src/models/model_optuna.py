import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
from src.utils.constants import GLOBAL_RANDOM_STATE, MODEL_REGISTRY


class OptunaModel:
    def __init__(self, model_type, target='HeartDisease', n_trials=50):
        self.model_type = model_type
        self.target = target
        self.n_trials = n_trials
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

    def _suggest_params(self, trial):
        if self.model_type == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
                'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),
                'random_state': GLOBAL_RANDOM_STATE
            }
        elif self.model_type == 'LogisticReg':
            return {
                'C': trial.suggest_float('C', 0.001, 10.0, log=True),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
                'random_state': GLOBAL_RANDOM_STATE
            }
        elif self.model_type == 'DecisionTree':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
                'random_state': GLOBAL_RANDOM_STATE
            }
        elif self.model_type == 'KNN':
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance'])
            }
        elif self.model_type == 'SVM':
            return {
                'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                'random_state': GLOBAL_RANDOM_STATE
            }
        elif self.model_type == 'GradientBoost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'random_state': GLOBAL_RANDOM_STATE
            }

    def train(self, data_path):
        df = pd.read_csv(data_path)
        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=GLOBAL_RANDOM_STATE
        )

        def objective(trial):
            params = self._suggest_params(trial)
            clf = MODEL_REGISTRY[self.model_type](**params)
            pipe = self._build_pipeline(clf)
            score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
            return score

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_STATE)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = study.best_params
        clf = MODEL_REGISTRY[self.model_type](**self.best_params)
        self.best_model = self._build_pipeline(clf)
        self.best_model.fit(X_train, y_train)

        test_score = self.best_model.score(X_test, y_test)
        print(f"{self.model_type} optimized accuracy: {test_score:.4f}")
        return test_score


def evaluate_models(dataset_path, target):
    results = []

    for name in MODEL_REGISTRY:
        model = OptunaModel(name, target)
        score = model.train(dataset_path)
        results.append((name, score))
        print(f"{name:15} Accuracy: {score:.4f}")

    return results


def save_results_to_csv(results, filename='model_performance_optuna.csv'):
    df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    results = evaluate_models('../datasets/heart.csv', target='HeartDisease')
    save_results_to_csv(results)