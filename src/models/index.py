import sys
import os
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.config import DATASET_CONFIG, OUTPUT_FILES, OPTIMIZATION_CONFIG
from src.utils.evaluator import evaluate_models, save_results_to_csv, compare_results

from src.models.model_base import BaseModel
from src.models.model_optuna import OptunaModel
from src.models.model_bayes import BayesOptModel
from src.models.model_hyp import HyperoptModel
from src.models.model_openbox import OpenboxModel
from src.models.model_skopt import SkoptModel


MODEL_CLASSES = {
    'base': BaseModel,
    'optuna': OptunaModel,
    'bayesopt': BayesOptModel,
    'hyperopt': HyperoptModel,
    'openbox': OpenboxModel,
    'skopt': SkoptModel
}


def run_single_method(method_name):
    if method_name not in MODEL_CLASSES:
        print(f"Unknown method: {method_name}")
        print(f"Available methods: {list(MODEL_CLASSES.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"RUNNING {method_name.upper()} OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Dataset: {DATASET_CONFIG['name']}")
    print(f"Target: {DATASET_CONFIG['target']}")
    print(f"Path: {DATASET_CONFIG['path']}")

    model_class = MODEL_CLASSES[method_name]
    config = OPTIMIZATION_CONFIG[method_name]

    results = evaluate_models(
        model_class=model_class,
        dataset_path=DATASET_CONFIG['path'],
        target=DATASET_CONFIG['target'],
        **config
    )

    save_results_to_csv(results, OUTPUT_FILES[method_name])
    return results


def run_all_methods():
    print(f"Running evaluation for dataset: {DATASET_CONFIG['name']}")
    print(f"Target variable: {DATASET_CONFIG['target']}")

    all_results = {}

    for method_name in MODEL_CLASSES.keys():
        try:
            results = run_single_method(method_name)
            all_results[method_name] = results
        except Exception as e:
            print(f"Error running {method_name}: {str(e)}")
            continue

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    compare_results(OUTPUT_FILES)


def main():
    parser = argparse.ArgumentParser(description='Run model evaluation with hyperparameter optimization')
    parser.add_argument(
        '--method',
        choices=list(MODEL_CLASSES.keys()) + ['all'],
        default='all',
        help='Optimization method to run (default: all)'
    )
    parser.add_argument(
        '--dataset',
        default=None,
        help='Override dataset path from config'
    )
    parser.add_argument(
        '--target',
        default=None,
        help='Override target variable from config'
    )

    args = parser.parse_args()

    if args.dataset:
        DATASET_CONFIG['path'] = args.dataset
    if args.target:
        DATASET_CONFIG['target'] = args.target

    if not os.path.exists(DATASET_CONFIG['path']):
        print(f"Error: Dataset not found at {DATASET_CONFIG['path']}")
        return

    if args.method == 'all':
        run_all_methods()
    else:
        run_single_method(args.method)


if __name__ == "__main__":
    main()