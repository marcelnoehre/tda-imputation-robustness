import gudhi
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from IPython.display import display, Markdown
from mdatagen.plots import PlotMissingData
from src.constants import *

def markdown(text):
    """
    Display text as Markdown.

    :param str text: The text to be displayed in Markdown format
    """
    display(Markdown(text))

def table(data, headers):
    """
    Display a table using Markdown.

    :param list[list] data: The data to be displayed in the table
    :param list[str] headers: The headers for the table
    """
    display(Markdown(tabulate(data, headers=headers, tablefmt='github')))

def plot_missing_data(missing_data, original_data, title=''):
    """
    Visualize the correlation of missing data using a heatmap.

    :param missing_data: Data with missing values
    :param original_data: Original data without missing values
    """
    miss_plot = PlotMissingData(data_missing=missing_data, data_original=original_data)
    miss_plot.visualize_miss('heatmap')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_rmse_vs_missingrate():
    """
    Plot RMSE against missing rates for different types of missingness.
    """
    df = pd.read_csv(RMSE_RESULTS)
    df = df[df[IMPUTATION_METHOD] == KNN]
    agg_df = df.groupby([MISSINGNESS_TYPE, MISSING_RATE, IMPUTATION_METHOD]).agg(
        rmse_mean=(RMSE, 'mean'), rmse_se=(RMSE, 'sem')
    ).reset_index()

    color_palette = sns.color_palette('Set2', n_colors=len(COLLECTIONS[IMPUTATION_METHOD]))
    colors = {
        mt: color_palette[i]
        for i, mt in enumerate(COLLECTIONS[MISSINGNESS_TYPE])
    }

    plt.figure(figsize=(10, 6))
    for mt in COLLECTIONS[MISSINGNESS_TYPE]:
        sub = agg_df[agg_df[MISSINGNESS_TYPE] == mt]
        plt.errorbar(
            sub[MISSING_RATE], sub['rmse_mean'], yerr=sub['rmse_se'],
            label=LABEL_SHORT[mt], color=colors[mt], marker='o', capsize=3, linewidth=1.8
        )

    plt.title(LABEL[RMSE])
    plt.xlabel(LABEL[MISSING_RATE])
    plt.ylabel(LABEL[RMSE])
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_distance_vs_missingrate_by_dim_and_type():
    """
    Plot the impact of missingness types and rates on various distance metrics across different dimensions.
    """
    df = pd.read_csv(IMPACT_MISSINGNESS)
    agg_df = df.groupby([MISSINGNESS_TYPE, MISSING_RATE, IMPUTATION_METHOD, TDA_METHOD, DIMENSION]).agg(
        wasserstein_distance_mean=(WS, 'mean'), wasserstein_distance_se=(WS, 'sem'),
        bottleneck_distance_mean=(BN, 'mean'), bottleneck_distance_se=(BN, 'sem'),
        l2_distance_landscape_mean=(L2PL, 'mean'), l2_distance_landscape_se=(L2PL, 'sem'),
        l2_distance_image_mean=(L2PI, 'mean'), l2_distance_image_se=(L2PI, 'sem'),
    ).reset_index()

    color_palette = sns.color_palette('Set1', n_colors=len(DIMENSIONS))
    colors = {
        dim: color_palette[i]
        for i, dim in enumerate(DIMENSIONS)
    }
    linestyle = {
        mt: ['-', '--', ':'][i % len(COLLECTIONS[MISSINGNESS_TYPE])]
        for i, mt in enumerate(COLLECTIONS[MISSINGNESS_TYPE])
    }

    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, metric in enumerate(COLLECTIONS[METRIC]):
        ax = axes[i]
        for _, dim in enumerate(DIMENSIONS):
            for mt_i, mt in enumerate(COLLECTIONS[MISSINGNESS_TYPE]):
                sub = agg_df[(agg_df[DIMENSION] == dim) & (agg_df[MISSINGNESS_TYPE] == mt)]
                if sub.empty:
                    continue
                ax.errorbar(
                    sub[MISSING_RATE], sub[f'{metric}_mean'], yerr=sub[f'{metric}_se'],
                    label=f'{LABEL_SHORT[DIMENSION]}={dim}, {LABEL_SHORT[mt]}', color=colors[dim], linestyle=linestyle[mt],
                    marker=['o', 's', '^'][mt_i % len(COLLECTIONS[MISSINGNESS_TYPE])], capsize=3, linewidth=1.8
                )
        ax.set_title(LABEL[metric])
        ax.set_xlabel(LABEL[MISSING_RATE])
        ax.set_ylabel(LABEL[metric])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_persistence_diagram(diagrams, title=''):
    max_dim = int(max(item[2] for item in diagrams[0]))
    persistence_intervals = [
        [item[0:2] for item in diagrams[0] if item[2] == dim]
        for dim in range(max_dim + 1)
    ]
    ax = gudhi.plot_persistence_diagram(persistence_intervals)
    ax.set_title(title)