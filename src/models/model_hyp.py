import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from hyperopt import tpe, fmin, Trials, STATUS_OK
from src.utils.constants import GLOBAL_RANDOM_STATE, HYPEROPT_SPACE, MODEL_REGISTRY

class HyperoptModel:
    def __init__(self, model_type, target='HeartDisease', max_evals=50):
        self.model_type = model_type
        self.target = target
        self.max_evals = max_evals
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=GLOBAL_RANDOM_STATE)

        def objective(params):
            clf = MODEL_REGISTRY[self.model_type](**params)
            pipe = self._build_pipeline(clf)
            score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()

        best = fmin(
            fn=objective,
            space=HYPEROPT_SPACE[self.model_type],
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.default_rng(GLOBAL_RANDOM_STATE)
        )

        if self.model_type == 'LogisticReg':
            solver_options = ['liblinear', 'lbfgs']
            best['solver'] = solver_options[best['solver']]
        elif self.model_type == 'SVM':
            kernel_options = ['linear', 'rbf', 'poly']
            best['kernel'] = kernel_options[best['kernel']]
        elif self.model_type == 'KNN':
            weights_options = ['uniform', 'distance']
            best['weights'] = weights_options[best['weights']]
            best['n_neighbors'] = list(range(3, 31))[best['n_neighbors']]
        elif self.model_type == 'GradientBoost':
            best['n_estimators'] = list(range(50, 201, 10))[best['n_estimators']]
            best['max_depth'] = list(range(3, 21))[best['max_depth']]
        elif self.model_type == 'RandomForest':
            best['n_estimators'] = list(range(50, 201, 10))[best['n_estimators']]
            best['max_depth'] = list(range(3, 21))[best['max_depth']]
        elif self.model_type == 'DecisionTree':
            best['max_depth'] = list(range(3, 21))[best['max_depth']]

        self.best_params = best
        clf = MODEL_REGISTRY[self.model_type](**best)
        self.best_model = self._build_pipeline(clf)
        self.best_model.fit(X_train, y_train)

        test_score = self.best_model.score(X_test, y_test)
        print(f"{self.model_type} optimized accuracy: {test_score:.4f}")
        return test_score

def evaluate_models(dataset_path, target):
    results = []

    for name in MODEL_REGISTRY:
        model = HyperoptModel(name, target)
        score = model.train(dataset_path)
        results.append((name, score))
        print(f"{name:15} Accuracy: {score:.4f}")

    return results

def save_results_to_csv(results, filename='model_performance_hyperopt.csv'):
    df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    results = evaluate_models('../datasets/heart.csv', target='HeartDisease')
    save_results_to_csv(results)