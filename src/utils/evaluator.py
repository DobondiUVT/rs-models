import pandas as pd
import os
from src.utils.config import MODEL_REGISTRY

def evaluate_models(model_class, dataset_path, target, **kwargs):
    results = []

    for model_name in MODEL_REGISTRY:
        print(f"Training {model_name}...")
        model = model_class(model_name, target, **kwargs)
        score = model.train(dataset_path)
        results.append((model_name, score))
        print(f"{model_name:15} Accuracy: {score:.4f}")

    return results


def save_results_to_csv(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def compare_results(result_files):
    comparison_data = {}

    for method, filepath in result_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            comparison_data[method] = df.set_index('Model')['Accuracy']

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + "="*60)
        print("COMPARISON OF OPTIMIZATION METHODS")
        print("="*60)
        print(comparison_df.round(4))
        print("\nBest performance per model:")
        for model in comparison_df.index:
            best_method = comparison_df.loc[model].idxmax()
            best_score = comparison_df.loc[model].max()
            print(f"{model:15}: {best_method:10} ({best_score:.4f})")

    return comparison_df if comparison_data else None