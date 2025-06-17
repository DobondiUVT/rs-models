import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config import MODEL_REGISTRY


def evaluate_models(model_class, dataset_path, target, **kwargs):
    results = []

    for model_name in MODEL_REGISTRY:
        print(f"Training {model_name}...")

        start_time = time.time()
        model = model_class(model_name, target, **kwargs)
        score = model.train(dataset_path)
        end_time = time.time()

        elapsed_time = end_time - start_time
        results.append((model_name, score, elapsed_time))
        print(f"{model_name:15} Accuracy: {score:.4f} | Time: {elapsed_time:.2f}s")

    return results


def save_results_to_csv(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Time_seconds'])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def create_comparison_chart(result_files, output_dir):
    comparison_accuracy = {}
    comparison_time = {}

    for method, filepath in result_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            comparison_accuracy[method] = df.set_index('Model')['Accuracy']
            comparison_time[method] = df.set_index('Model')['Time_seconds']

    if not comparison_accuracy:
        return None

    accuracy_df = pd.DataFrame(comparison_accuracy)
    time_df = pd.DataFrame(comparison_time)

    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Accuracy heatmap
    sns.heatmap(accuracy_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1, cbar_kws={'label': 'Accuracy'})
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Optimization Method')
    ax1.set_ylabel('Model')

    # Time heatmap
    sns.heatmap(time_df, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax2, cbar_kws={'label': 'Time (seconds)'})
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Optimization Method')
    ax2.set_ylabel('Model')

    # Best accuracy per model
    best_accuracy = accuracy_df.max(axis=1)
    best_method = accuracy_df.idxmax(axis=1)
    colors = plt.cm.Set3(range(len(best_accuracy)))
    bars = ax3.bar(range(len(best_accuracy)), best_accuracy, color=colors)
    ax3.set_title('Best Accuracy per Model', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Best Accuracy')
    ax3.set_xticks(range(len(best_accuracy)))
    ax3.set_xticklabels(best_accuracy.index, rotation=45)

    for i, (bar, method) in enumerate(zip(bars, best_method)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{method}\n{height:.3f}', ha='center', va='bottom', fontsize=8)

    # Accuracy vs Time scatter
    for method in accuracy_df.columns:
        x = time_df[method]
        y = accuracy_df[method]
        ax4.scatter(x, y, label=method, s=100, alpha=0.7)

        for i, model in enumerate(accuracy_df.index):
            ax4.annotate(model, (x.iloc[i], y.iloc[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax4.set_title('Accuracy vs Training Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Training Time (seconds)')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    chart_path = f"{output_dir}/model_comparison_chart.png"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison chart saved to {chart_path}")
    return chart_path


def compare_results(result_files):
    comparison_accuracy = {}
    comparison_time = {}

    for method, filepath in result_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            comparison_accuracy[method] = df.set_index('Model')['Accuracy']
            comparison_time[method] = df.set_index('Model')['Time_seconds']

    if comparison_accuracy:
        accuracy_df = pd.DataFrame(comparison_accuracy)
        time_df = pd.DataFrame(comparison_time)

        print("\n" + "="*80)
        print("ACCURACY COMPARISON")
        print("="*80)
        print(accuracy_df.round(4))

        print("\n" + "="*80)
        print("TIME COMPARISON (seconds)")
        print("="*80)
        print(time_df.round(2))

        print("\n" + "="*80)
        print("BEST PERFORMANCE PER MODEL")
        print("="*80)
        for model in accuracy_df.index:
            best_method = accuracy_df.loc[model].idxmax()
            best_score = accuracy_df.loc[model].max()
            best_time = time_df.loc[model, best_method]
            print(f"{model:15}: {best_method:10} (Acc: {best_score:.4f}, Time: {best_time:.2f}s)")

        print("\n" + "="*80)
        print("EFFICIENCY ANALYSIS (Accuracy/Time)")
        print("="*80)
        efficiency_df = accuracy_df / time_df
        for model in efficiency_df.index:
            best_method = efficiency_df.loc[model].idxmax()
            best_efficiency = efficiency_df.loc[model].max()
            print(f"{model:15}: {best_method:10} (Efficiency: {best_efficiency:.4f})")

        output_dir = os.path.dirname(list(result_files.values())[0])
        create_comparison_chart(result_files, output_dir)

        return accuracy_df, time_df

    return None, None