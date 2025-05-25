import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from bayes_opt import BayesianOptimization
from src.utils.constants import GLOBAL_RANDOM_STATE, BAYESOPT_SPACE, MODEL_REGISTRY


class BayesOptModel:
    def __init__(self, model_type, target='HeartDisease', n_iter=50):
        self.model_type = model_type
        self.target = target
        self.n_iter = n_iter
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

    def _create_objective(self, X_train, y_train):
        def objective(**kwargs):
            params = {}

            if self.model_type == 'RandomForest':
                params = {
                    'n_estimators': int(kwargs['n_estimators']),
                    'max_depth': int(kwargs['max_depth']),
                    'min_samples_split': kwargs['min_samples_split'],
                    'min_samples_leaf': kwargs['min_samples_leaf'],
                    'random_state': GLOBAL_RANDOM_STATE
                }
            elif self.model_type == 'LogisticReg':
                params = {
                    'C': kwargs['C'],
                    'solver': ['liblinear', 'lbfgs'][int(kwargs['solver'])],
                    'random_state': GLOBAL_RANDOM_STATE
                }
            elif self.model_type == 'DecisionTree':
                params = {
                    'max_depth': int(kwargs['max_depth']),
                    'min_samples_split': kwargs['min_samples_split'],
                    'random_state': GLOBAL_RANDOM_STATE
                }
            elif self.model_type == 'KNN':
                params = {
                    'n_neighbors': int(kwargs['n_neighbors']),
                    'weights': ['uniform', 'distance'][int(kwargs['weights'])]
                }
            elif self.model_type == 'SVM':
                params = {
                    'C': kwargs['C'],
                    'kernel': ['linear', 'rbf', 'poly'][int(kwargs['kernel'])],
                    'random_state': GLOBAL_RANDOM_STATE
                }
            elif self.model_type == 'GradientBoost':
                params = {
                    'n_estimators': int(kwargs['n_estimators']),
                    'learning_rate': kwargs['learning_rate'],
                    'max_depth': int(kwargs['max_depth']),
                    'random_state': GLOBAL_RANDOM_STATE
                }

            clf = MODEL_REGISTRY[self.model_type](**params)
            pipe = self._build_pipeline(clf)
            score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
            return score

        return objective

    def train(self, data_path):
        df = pd.read_csv(data_path)
        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=GLOBAL_RANDOM_STATE
        )

        objective = self._create_objective(X_train, y_train)
        bounds = BAYESOPT_SPACE[self.model_type]

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=bounds,
            random_state=GLOBAL_RANDOM_STATE,
            verbose=0
        )

        optimizer.maximize(
            init_points=10,
            n_iter=self.n_iter
        )

        best_params = optimizer.max['params']

        # Convert parameters back to proper types/values
        if self.model_type == 'RandomForest':
            self.best_params = {
                'n_estimators': int(best_params['n_estimators']),
                'max_depth': int(best_params['max_depth']),
                'min_samples_split': best_params['min_samples_split'],
                'min_samples_leaf': best_params['min_samples_leaf'],
                'random_state': GLOBAL_RANDOM_STATE
            }
        elif self.model_type == 'LogisticReg':
            self.best_params = {
                'C': best_params['C'],
                'solver': ['liblinear', 'lbfgs'][int(best_params['solver'])],
                'random_state': GLOBAL_RANDOM_STATE
            }
        elif self.model_type == 'DecisionTree':
            self.best_params = {
                'max_depth': int(best_params['max_depth']),
                'min_samples_split': best_params['min_samples_split'],
                'random_state': GLOBAL_RANDOM_STATE
            }
        elif self.model_type == 'KNN':
            self.best_params = {
                'n_neighbors': int(best_params['n_neighbors']),
                'weights': ['uniform', 'distance'][int(best_params['weights'])]
            }
        elif self.model_type == 'SVM':
            self.best_params = {
                'C': best_params['C'],
                'kernel': ['linear', 'rbf', 'poly'][int(best_params['kernel'])],
                'random_state': GLOBAL_RANDOM_STATE
            }
        elif self.model_type == 'GradientBoost':
            self.best_params = {
                'n_estimators': int(best_params['n_estimators']),
                'learning_rate': best_params['learning_rate'],
                'max_depth': int(best_params['max_depth']),
                'random_state': GLOBAL_RANDOM_STATE
            }

        clf = MODEL_REGISTRY[self.model_type](**self.best_params)
        self.best_model = self._build_pipeline(clf)
        self.best_model.fit(X_train, y_train)

        test_score = self.best_model.score(X_test, y_test)
        print(f"{self.model_type} optimized accuracy: {test_score:.4f}")
        return test_score


def evaluate_models(dataset_path, target):
    results = []

    for name in MODEL_REGISTRY:
        model = BayesOptModel(name, target)
        score = model.train(dataset_path)
        results.append((name, score))
        print(f"{name:15} Accuracy: {score:.4f}")

    return results


def save_results_to_csv(results, filename='model_performance_bayesopt.csv'):
    df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    results = evaluate_models('../datasets/heart.csv', target='HeartDisease')
    save_results_to_csv(results)