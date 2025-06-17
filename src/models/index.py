import sys
import os
import argparse
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.utils.config import (
    DATASET_CONFIGS,
    OPTIMIZATION_CONFIG,
    get_output_files
)
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


def run_single_method(method_name, dataset_config):
    if method_name not in MODEL_CLASSES:
        print(f"Unknown method: {method_name}")
        print(f"Available methods: {list(MODEL_CLASSES.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"RUNNING {method_name.upper()} OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_config['name']}")
    print(f"Target: {dataset_config['target']}")
    print(f"Path: {dataset_config['path']}")

    model_class = MODEL_CLASSES[method_name]
    config = OPTIMIZATION_CONFIG[method_name]
    
    output_files = get_output_files(dataset_config['name'])

    results = evaluate_models(
        model_class=model_class,
        dataset_path=dataset_config['path'],
        target=dataset_config['target'],
        **config
    )

    save_results_to_csv(results, output_files[method_name])
    return results


def run_all_methods(dataset_config):
    print(f"Running evaluation for dataset: {dataset_config['name']}")
    print(f"Target variable: {dataset_config['target']}")

    all_results = {}
    output_files = get_output_files(dataset_config['name'])

    for method_name in MODEL_CLASSES.keys():
        try:
            results = run_single_method(method_name, dataset_config)
            all_results[method_name] = results
        except Exception as e:
            print(f"Error running {method_name}: {str(e)}")
            continue

    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON FOR {dataset_config['name']}")
    print(f"{'='*60}")
    compare_results(output_files)
    return all_results


def run_all_datasets(method=None):
    all_dataset_results = {}
    
    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        print(f"\n{'#'*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'#'*80}")
        
        if not os.path.exists(dataset_config['path']):
            print(f"Warning: Dataset not found at {dataset_config['path']}, skipping...")
            continue
            
        try:
            if method and method != 'all':
                results = run_single_method(method, dataset_config)
            else:
                results = run_all_methods(dataset_config)
            all_dataset_results[dataset_name] = results
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
            continue
    
    print("\nAll datasets processed!")
    return all_dataset_results


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
        choices=list(DATASET_CONFIGS.keys()) + ['all'],
        default='all',
        help='Dataset to process (default: all)'
    )
    parser.add_argument(
        '--target',
        default=None,
        help='Override target variable from config'
    )

    args = parser.parse_args()

    if args.dataset == 'all':
        run_all_datasets(args.method)
    else:
        dataset_config = DATASET_CONFIGS[args.dataset].copy()
        if args.target:
            dataset_config['target'] = args.target
            
        if not os.path.exists(dataset_config['path']):
            print(f"Error: Dataset not found at {dataset_config['path']}")
            return

        if args.method == 'all':
            run_all_methods(dataset_config)
        else:
            run_single_method(args.method, dataset_config)


if __name__ == "__main__":
    main()