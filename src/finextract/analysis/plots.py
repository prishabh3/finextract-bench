from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_accuracy_comparison(metrics_dict: dict[str, Any], output_path: Path) -> None:
    """Plots exact vs 1% vs 5% accuracy across pipelines as a grouped bar chart."""
    data = []
    for pipeline_name, metrics in metrics_dict.items():
        # Handle both object attribute and dict key access for flexibility
        if hasattr(metrics, 'exact_accuracy'):
            exact_acc = metrics.exact_accuracy
            acc_1pct = metrics.accuracy_1pct
            acc_5pct = metrics.accuracy_5pct
        else:
            exact_acc = metrics.get('exact_accuracy', 0.0)
            acc_1pct = metrics.get('accuracy_1pct', 0.0)
            acc_5pct = metrics.get('accuracy_5pct', 0.0)

        data.append({'Pipeline': pipeline_name, 'Accuracy Type': 'Exact', 'Value': exact_acc})
        data.append({'Pipeline': pipeline_name, 'Accuracy Type': '1% Margin', 'Value': acc_1pct})
        data.append({'Pipeline': pipeline_name, 'Accuracy Type': '5% Margin', 'Value': acc_5pct})

    if not data:
        return

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Pipeline', y='Value', hue='Accuracy Type')
    plt.title('Accuracy Comparison Across Pipelines')
    plt.ylabel('Accuracy Ratio')
    plt.ylim(0, 1)
    plt.legend(title='Accuracy Type')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_failure_distribution(failures: list[dict], output_path: Path) -> None:
    """Plots a horizontal bar chart of failure types."""
    if not failures:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, 'No Failures', horizontalalignment='center', verticalalignment='center')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        return

    failure_counts = {}
    for failure in failures:
        if hasattr(failure, 'failure_type'):
            error_type = failure.failure_type
        else:
            error_type = failure.get('failure_type', failure.get('error_type', 'Unknown Error'))

        failure_counts[error_type] = failure_counts.get(error_type, 0) + 1

    df = pd.DataFrame(list(failure_counts.items()), columns=['Failure Type', 'Count'])
    df = df.sort_values('Count', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, y='Failure Type', x='Count', orient='h')
    plt.title('Distribution of Failure Types')
    plt.xlabel('Number of Occurrences')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_latency_vs_accuracy(metrics_dict: dict[str, Any], output_path: Path) -> None:
    """Scatter plot of mean latency vs 1% accuracy for each pipeline."""
    data = []
    for pipeline_name, metrics in metrics_dict.items():
        if hasattr(metrics, 'mean_latency_ms'):
            mean_lat_ms = metrics.mean_latency_ms or 0.0
            acc_1pct = metrics.accuracy_1pct
        else:
            mean_lat_ms = metrics.get('mean_latency_ms', 0.0) or 0.0
            acc_1pct = metrics.get('accuracy_1pct', 0.0)

        mean_lat_s = mean_lat_ms / 1000.0
        data.append({
            'Pipeline': pipeline_name,
            'Mean Latency (s)': mean_lat_s,
            '1% Accuracy': acc_1pct
        })

    if not data:
        return

    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Mean Latency (s)', y='1% Accuracy', hue='Pipeline', s=100)

    for i in range(df.shape[0]):
        plt.text(df['Mean Latency (s)'][i], df['1% Accuracy'][i],
                 f" {df['Pipeline'][i]}",
                 horizontalalignment='left', size='medium', color='black')

    plt.title('Latency vs Accuracy Trade-off')
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
