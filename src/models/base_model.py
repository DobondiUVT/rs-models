import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from src.utils.config import MODEL_REGISTRY, GLOBAL_RANDOM_STATE, TRAIN_CONFIG

class BaseModelClass:
    def __init__(self, model_type, target):
        self.model_type = model_type
        self.target = target
        self.best_model = None
        self.best_params = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

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

    def _load_and_split_data(self, data_path):
        df = pd.read_csv(data_path)
        X = df.drop(columns=[self.target])
        y = df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=TRAIN_CONFIG['test_size'],
            random_state=GLOBAL_RANDOM_STATE
        )

    def _evaluate_model(self, pipeline):
        return cross_val_score(
            pipeline,
            self.X_train,
            self.y_train,
            cv=TRAIN_CONFIG['cv_folds'],
            scoring=TRAIN_CONFIG['scoring']
        ).mean()

    def _fit_best_model(self, best_params):
        classifier = MODEL_REGISTRY[self.model_type](**best_params)
        self.best_model = self._build_pipeline(classifier)
        self.best_model.fit(self.X_train, self.y_train)
        return self.best_model.score(self.X_test, self.y_test)

    def train(self, data_path):
        raise NotImplementedError("Subclasses must implement train method")