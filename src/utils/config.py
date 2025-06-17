from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import hp
from skopt.space import Real, Integer, Categorical
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
import numpy as np
import datetime
import os

GLOBAL_RANDOM_STATE = 42

DATASET_CONFIGS = {
    'heart': {
        'name': 'heart',
        'path': 'src/datasets/heart.csv',
        'target': 'HeartDisease'
    },

    'wine_quality_red': {
        'name': 'wine_quality_red',
        'path': 'src/datasets/winequality-red.csv',
        'target': 'quality'
    },

    'diabetes': {
        'name': 'diabetes',
        'path': 'src/datasets/diabetes.csv',
        'target': 'Outcome'
    },

    'iris': {
        'name': 'iris',
        'path': 'src/datasets/iris.csv',
        'target': 'species',
        'url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
    },

    'breast_cancer': {
        'name': 'breast_cancer',
        'path': 'src/datasets/breast-cancer.csv',
        'target': 'diagnosis',
        'url': 'https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv'
    },
}

def get_results_dir(dataset_name):
    base_dir = 'src/results'
    dataset_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir

def get_output_files(dataset_name):
    results_dir = get_results_dir(dataset_name)
    return {
        'base': f'{results_dir}/model_performance_base.csv',
        'optuna': f'{results_dir}/model_performance_optuna.csv',
        'bayesopt': f'{results_dir}/model_performance_bayesopt.csv',
        'hyperopt': f'{results_dir}/model_performance_hyperopt.csv',
        'openbox': f'{results_dir}/model_performance_openbox.csv',
        'skopt': f'{results_dir}/model_performance_skopt.csv'
    }

DATASETS_DIR = 'src/datasets'

MODEL_REGISTRY = {
    'RandomForest': RandomForestClassifier,
    'LogisticReg': LogisticRegression,
    'DecisionTree': DecisionTreeClassifier,
    'SVM': SVC,
    'KNN': KNeighborsClassifier
}

TRAIN_CONFIG = {
    'test_size': 0.2,
    'cv_folds': 5,
    'scoring': 'accuracy'
}

OPTIMIZATION_CONFIG = {
    'base': {},
    'optuna': {'n_trials': 5},
    'bayesopt': {'n_iter': 5},
    'hyperopt': {'max_evals': 5},
    'openbox': {'max_runs': 5},
    'skopt': {'n_calls': 5}
}

DATASET_CONFIG = DATASET_CONFIGS['wine_quality_red']

RESULTS_DIR = get_results_dir(DATASET_CONFIG['name'])

MODEL_DEFAULT_PARAMS = {
    'RandomForest': {'random_state': GLOBAL_RANDOM_STATE},
    'LogisticReg': {'random_state': GLOBAL_RANDOM_STATE},
    'DecisionTree': {'random_state': GLOBAL_RANDOM_STATE},
    'SVM': {'random_state': GLOBAL_RANDOM_STATE},
    'KNN': {}
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
    ]
}

OPENBOX_SPACE = {
    'RandomForest': ConfigurationSpace(),
    'LogisticReg': ConfigurationSpace(),
    'DecisionTree': ConfigurationSpace(),
    'KNN': ConfigurationSpace(),
    'SVM': ConfigurationSpace()
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
    'RandomForest': {
        'n_estimators': list(range(50, 201, 10)),
        'max_depth': list(range(3, 21))
    },
    'DecisionTree': {
        'max_depth': list(range(3, 21))
    }
}