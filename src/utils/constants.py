from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import hp
from skopt.space import Real, Integer, Categorical
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
import numpy as np

GLOBAL_RANDOM_STATE = 42

MODEL_REGISTRY = {
    'RandomForest': RandomForestClassifier,
    'LogisticReg': LogisticRegression,
    'DecisionTree': DecisionTreeClassifier,
    'SVM': SVC,
    'KNN': KNeighborsClassifier,
    'GradientBoost': GradientBoostingClassifier
}

MODEL_DEFAULT_PARAMS = {
    'RandomForest': {'random_state': GLOBAL_RANDOM_STATE},
    'LogisticReg': {'random_state': GLOBAL_RANDOM_STATE},
    'DecisionTree': {'random_state': GLOBAL_RANDOM_STATE},
    'SVM': {'random_state': GLOBAL_RANDOM_STATE},
    'KNN': {},
    'GradientBoost': {'random_state': GLOBAL_RANDOM_STATE}
}

HYPEROPT_SPACE = {
    'RandomForest': {
        'n_estimators': hp.choice('n_estimators', range(50, 201, 10)),
        'max_depth': hp.choice('max_depth', range(3, 21)),
        'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0.1, 0.5)
    },
    'LogisticReg': {
        'C': hp.loguniform('C', np.log(0.001), np.log(10)),
        'solver': hp.choice('solver', ['liblinear', 'lbfgs'])
    },
    'DecisionTree': {
        'max_depth': hp.choice('max_depth', range(3, 21)),
        'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0)
    },
    'KNN': {
        'n_neighbors': hp.choice('n_neighbors', range(3, 31)),
        'weights': hp.choice('weights', ['uniform', 'distance'])
    },
    'SVM': {
        'C': hp.loguniform('C', np.log(0.01), np.log(10)),
        'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly'])
    },
    'GradientBoost': {
        'n_estimators': hp.choice('n_estimators', range(50, 201, 10)),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'max_depth': hp.choice('max_depth', range(3, 21))
    }
}

SKOPT_SPACE = {
    'RandomForest': [
        Integer(50, 200, name='n_estimators'),
        Integer(3, 20, name='max_depth'),
        Real(0.1, 1.0, name='min_samples_split'),
        Real(0.1, 0.5, name='min_samples_leaf'),
    ],
    'LogisticReg': [
        Real(0.001, 10.0, prior='log-uniform', name='C'),
        Categorical(['liblinear', 'lbfgs'], name='solver'),
    ],
    'DecisionTree': [
        Integer(3, 20, name='max_depth'),
        Real(0.1, 1.0, name='min_samples_split'),
    ],
    'KNN': [
        Integer(3, 30, name='n_neighbors'),
        Categorical(['uniform', 'distance'], name='weights')
    ],
    'SVM': [
        Real(0.01, 10.0, prior='log-uniform', name='C'),
        Categorical(['linear', 'rbf', 'poly'], name='kernel'),
    ],
    'GradientBoost': [
        Integer(50, 200, name='n_estimators'),
        Real(0.01, 0.3, name='learning_rate'),
        Integer(3, 20, name='max_depth'),
    ]
}

OPENBOX_SPACE = {
    'RandomForest': ConfigurationSpace(),
    'LogisticReg': ConfigurationSpace(),
    'DecisionTree': ConfigurationSpace(),
    'KNN': ConfigurationSpace(),
    'SVM': ConfigurationSpace(),
    'GradientBoost': ConfigurationSpace()
}

# Initialize OpenBox spaces
OPENBOX_SPACE['RandomForest'].add_hyperparameters([
    UniformIntegerHyperparameter('n_estimators', 50, 200),
    UniformIntegerHyperparameter('max_depth', 3, 20),
    UniformFloatHyperparameter('min_samples_split', 0.1, 1.0),
    UniformFloatHyperparameter('min_samples_leaf', 0.1, 0.5),
])

OPENBOX_SPACE['LogisticReg'].add_hyperparameters([
    UniformFloatHyperparameter('C', 0.001, 10.0, log=True),
    CategoricalHyperparameter('solver', ['liblinear', 'lbfgs']),
])

OPENBOX_SPACE['DecisionTree'].add_hyperparameters([
    UniformIntegerHyperparameter('max_depth', 3, 20),
    UniformFloatHyperparameter('min_samples_split', 0.1, 1.0),
])

OPENBOX_SPACE['KNN'].add_hyperparameters([
    UniformIntegerHyperparameter('n_neighbors', 3, 30),
    CategoricalHyperparameter('weights', ['uniform', 'distance'])
])

OPENBOX_SPACE['SVM'].add_hyperparameters([
    UniformFloatHyperparameter('C', 0.01, 10.0, log=True),
    CategoricalHyperparameter('kernel', ['linear', 'rbf', 'poly']),
])

OPENBOX_SPACE['GradientBoost'].add_hyperparameters([
    UniformIntegerHyperparameter('n_estimators', 50, 200),
    UniformFloatHyperparameter('learning_rate', 0.01, 0.3),
    UniformIntegerHyperparameter('max_depth', 3, 20),
])

BAYESOPT_SPACE = {
    'RandomForest': {
        'n_estimators': (50, 200),
        'max_depth': (3, 20),
        'min_samples_split': (0.1, 1.0),
        'min_samples_leaf': (0.1, 0.5)
    },
    'LogisticReg': {
        'C': (0.001, 10.0),
        'solver': (0, 1)
    },
    'DecisionTree': {
        'max_depth': (3, 20),
        'min_samples_split': (0.1, 1.0)
    },
    'KNN': {
        'n_neighbors': (3, 30),
        'weights': (0, 1)
    },
    'SVM': {
        'C': (0.01, 10.0),
        'kernel': (0, 2)
    },
    'GradientBoost': {
        'n_estimators': (50, 200),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 20)
    }
}