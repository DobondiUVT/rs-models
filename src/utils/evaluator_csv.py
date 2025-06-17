import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def find_all_result_csvs(base_dir='src/results'):
    """Find all CSV files in the results directory and its subdirectories."""
    csv_files = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith('model_performance_') and file.endswith('.csv'):
                method = file.replace('model_performance_', '').replace('.csv', '')
                full_path = os.path.join(root, file)
                dataset = os.path.basename(os.path.dirname(full_path))
                if dataset not in csv_files:
                    csv_files[dataset] = {}
                csv_files[dataset][method] = full_path
    return csv_files

def sort_methods(df):
    """Ensure 'base' is the first method in the DataFrame."""
    if 'base' in df.columns:
        cols = ['base'] + [col for col in sorted(df.columns) if col != 'base']
        return df[cols]
    return df

def create_accuracy_heatmap(accuracy_df, output_path):
    """Create and save accuracy heatmap."""
    accuracy_df = sort_methods(accuracy_df)
    plt.figure(figsize=(10, 6))
    sns.heatmap(accuracy_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Accuracy'})
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Optimization Method')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_time_heatmap(time_df, output_path):
    """Create and save time heatmap."""
    time_df = sort_methods(time_df)
    plt.figure(figsize=(10, 6))
    sns.heatmap(time_df, annot=True, fmt='.1f', cmap='RdYlBu_r',
                cbar_kws={'label': 'Time (seconds)'})
    plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Optimization Method')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_model_abbreviation(model_name):
    """Convert model names to shorter abbreviations."""
    abbreviations = {
        'LogisticReg': 'LR',
        'RandomForest': 'RF',
        'DecisionTree': 'DT',
        'SVM': 'SVM',  # already short
        'KNN': 'KNN'   # already short
    }
    return abbreviations.get(model_name, model_name)

def create_best_accuracy_bar(accuracy_df, output_path):
    """Create and save best accuracy bar chart."""
    plt.figure(figsize=(10, 6))
    best_accuracy = accuracy_df.max(axis=1)
    best_method = accuracy_df.idxmax(axis=1)
    colors = plt.cm.Set3(range(len(best_accuracy)))
    
    # Add padding to y-axis limits
    y_min = min(best_accuracy) * 0.95  # 5% padding below
    y_max = max(best_accuracy) * 1.05  # 5% padding above
    
    bars = plt.bar(range(len(best_accuracy)), best_accuracy, color=colors)
    plt.title('Best Accuracy per Model', fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Best Accuracy')
    plt.xticks(range(len(best_accuracy)), 
               [get_model_abbreviation(model) for model in best_accuracy.index], 
               rotation=45)
    plt.ylim(y_min, y_max)

    for i, (bar, method) in enumerate(zip(bars, best_method)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (y_max - y_min) * 0.01,
                f'{method}\n{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_accuracy_time_scatter(accuracy_df, time_df, output_path):
    """Create and save accuracy vs time scatter plot."""
    accuracy_df = sort_methods(accuracy_df)
    time_df = sort_methods(time_df)
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    for method in accuracy_df.columns:
        x = time_df[method]
        y = accuracy_df[method]
        plt.scatter(x, y, label=method, s=100, alpha=0.7)
        
        # Add labels directly next to points
        for i, model in enumerate(accuracy_df.index):
            plt.annotate(get_model_abbreviation(model), 
                        (x.iloc[i], y.iloc[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)

    plt.title('Accuracy vs Training Time', fontsize=14, fontweight='bold')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_dataset_results(dataset_name, result_files):
    """Process results for a single dataset and create charts."""
    comparison_accuracy = {}
    comparison_time = {}

    # Read data from CSV files
    for method, filepath in result_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            comparison_accuracy[method] = df.set_index('Model')['Accuracy']
            comparison_time[method] = df.set_index('Model')['Time_seconds']

    if not comparison_accuracy:
        print(f"No data found for dataset: {dataset_name}")
        return

    # Create DataFrames
    accuracy_df = pd.DataFrame(comparison_accuracy)
    time_df = pd.DataFrame(comparison_time)

    # Create output directory
    output_dir = f"src/results/charts/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Create individual charts
    create_accuracy_heatmap(accuracy_df, f"{output_dir}/accuracy_heatmap.png")
    create_time_heatmap(time_df, f"{output_dir}/time_heatmap.png")
    create_best_accuracy_bar(accuracy_df, f"{output_dir}/best_accuracy_bar.png")
    create_accuracy_time_scatter(accuracy_df, time_df, f"{output_dir}/accuracy_time_scatter.png")

    print(f"Charts created for dataset: {dataset_name}")

def main():
    # Find all CSV files
    result_files = find_all_result_csvs()
    
    if not result_files:
        print("No result CSV files found!")
        return

    # Process each dataset
    for dataset_name, dataset_results in result_files.items():
        print(f"\nProcessing dataset: {dataset_name}")
        process_dataset_results(dataset_name, dataset_results)

if __name__ == "__main__":
    main() 