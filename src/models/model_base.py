import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from src.utils.constants import MODEL_REGISTRY, MODEL_DEFAULT_PARAMS, GLOBAL_RANDOM_STATE


class BaseModel:
    def __init__(self, model_type, target='HeartDisease'):
        self.target = target
        self.model = self._build_pipeline(MODEL_REGISTRY[model_type](**MODEL_DEFAULT_PARAMS[model_type]))

    def _build_pipeline(self, classifier):
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
            ('classifier', classifier)
        ])

    def train(self, data_path, test_size=0.2):
        df = pd.read_csv(data_path)
        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=GLOBAL_RANDOM_STATE
        )

        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)


def evaluate_models(dataset_path, target):
    results = []

    for name in MODEL_REGISTRY:
        model = BaseModel(name, target)
        score = model.train(dataset_path)
        results.append((name, score))
        print(f"{name:15} Accuracy: {score:.4f}")

    return results


def save_results_to_csv(results, filename='model_performance.csv'):
    df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    results = evaluate_models('../datasets/heart.csv', target='HeartDisease')
    save_results_to_csv(results)
