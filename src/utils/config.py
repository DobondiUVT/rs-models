from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import hp
from skopt.space import Real, Integer, Categorical
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
import numpy as np
import datetime

GLOBAL_RANDOM_STATE = 42

DATASET_CONFIG = {
    'name': 'heart',
    'path': '../datasets/heart.csv',
    'target': 'HeartDisease'
}

RESULTS_DIR = f'../results/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}/{DATASET_CONFIG["name"]}'
DATASETS_DIR = '../datasets'

MODEL_REGISTRY = {
    'RandomForest': RandomForestClassifier,
    'LogisticReg': LogisticRegression,
    'DecisionTree': DecisionTreeClassifier,
    'SVM': SVC,
    'KNN': KNeighborsClassifier,
    'GradientBoost': GradientBoostingClassifier
}

TRAIN_CONFIG = {
    'test_size': 0.2,
    'cv_folds': 5,
    'scoring': 'accuracy'
}

OPTIMIZATION_CONFIG = {
    'base': {},
    'optuna': {'n_trials': 100},
    'bayesopt': {'n_iter': 100},
    'hyperopt': {'max_evals': 100},
    'openbox': {'max_runs': 100},
    'skopt': {'n_calls': 100}
}

OUTPUT_FILES = {
    'base': f'{RESULTS_DIR}/model_performance_base.csv',
    'optuna': f'{RESULTS_DIR}/model_performance_optuna.csv',
    'bayesopt': f'{RESULTS_DIR}/model_performance_bayesopt.csv',
    'hyperopt': f'{RESULTS_DIR}/model_performance_hyperopt.csv',
    'openbox': f'{RESULTS_DIR}/model_performance_openbox.csv',
    'skopt': f'{RESULTS_DIR}/model_performance_skopt.csv'
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

OPTUNA_PARAMS = {
    'RandomForest': {
        'n_estimators': ('int', 50, 200),
        'max_depth': ('int', 3, 20),
        'min_samples_split': ('float', 0.1, 1.0),
        'min_samples_leaf': ('float', 0.1, 0.5),
        'random_state': GLOBAL_RANDOM_STATE
    },
    'LogisticReg': {
        'C': ('float_log', 0.001, 10.0),
        'solver': ('categorical', ['liblinear', 'lbfgs']),
        'random_state': GLOBAL_RANDOM_STATE
    },
    'DecisionTree': {
        'max_depth': ('int', 3, 20),
        'min_samples_split': ('float', 0.1, 1.0),
        'random_state': GLOBAL_RANDOM_STATE
    },
    'KNN': {
        'n_neighbors': ('int', 3, 30),
        'weights': ('categorical', ['uniform', 'distance'])
    },
    'SVM': {
        'C': ('float_log', 0.01, 10.0),
        'kernel': ('categorical', ['linear', 'rbf', 'poly']),
        'random_state': GLOBAL_RANDOM_STATE
    },
    'GradientBoost': {
        'n_estimators': ('int', 50, 200),
        'learning_rate': ('float', 0.01, 0.3),
        'max_depth': ('int', 3, 20),
        'random_state': GLOBAL_RANDOM_STATE
    }
}

BAYESOPT_CONVERSION = {
    'RandomForest': {
        'n_estimators': 'int',
        'max_depth': 'int',
        'min_samples_split': 'float',
        'min_samples_leaf': 'float',
        'random_state': GLOBAL_RANDOM_STATE
    },
    'LogisticReg': {
        'C': 'float',
        'solver': (['liblinear', 'lbfgs'], 'int'),
        'random_state': GLOBAL_RANDOM_STATE
    },
    'DecisionTree': {
        'max_depth': 'int',
        'min_samples_split': 'float',
        'random_state': GLOBAL_RANDOM_STATE
    },
    'KNN': {
        'n_neighbors': 'int',
        'weights': (['uniform', 'distance'], 'int')
    },
    'SVM': {
        'C': 'float',
        'kernel': (['linear', 'rbf', 'poly'], 'int'),
        'random_state': GLOBAL_RANDOM_STATE
    },
    'GradientBoost': {
        'n_estimators': 'int',
        'learning_rate': 'float',
        'max_depth': 'int',
        'random_state': GLOBAL_RANDOM_STATE
    }
}

HYPEROPT_CONVERSION = {
    'LogisticReg': {
        'solver': ['liblinear', 'lbfgs']
    },
    'SVM': {
        'kernel': ['linear', 'rbf', 'poly']
    },
    'KNN': {
        'weights': ['uniform', 'distance'],
        'n_neighbors': list(range(3, 31))
    },
    'GradientBoost': {
        'n_estimators': list(range(50, 201, 10)),
        'max_depth': list(range(3, 21))
    },
    'RandomForest': {
        'n_estimators': list(range(50, 201, 10)),
        'max_depth': list(range(3, 21))
    },
    'DecisionTree': {
        'max_depth': list(range(3, 21))
    }
}