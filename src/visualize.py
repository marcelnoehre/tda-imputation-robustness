import gudhi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate
from IPython.display import display, Markdown
from mdatagen.plots import PlotMissingData
from src.constants import *
from src.missingness import MISSINGNESS
from src.imputation import IMPUTATION
from src.tda import TDA
from src.normalize import normalize_by_diameter

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

def setup_figure(cells=None, rows=None, cols=None):
    if cells:
        rows, cols = int(cells * 0.5), int(cells * 0.5)
    return plt.subplots(rows, cols, figsize=(ASPECT * HEIGHT * cols, HEIGHT * rows), sharex=False, sharey=False)

def group_results(csv, group, metrics, filter=None, zeros=False):
    df = pd.read_csv(csv)
    if filter is not None:
        df = df[df[filter[0]] == filter[1]]
    
    if zeros:
        new_rows = []
        for dataset in df[DATASET].unique():
            for mt in df[MISSINGNESS_TYPE].unique():
                for imp in df[IMPUTATION_METHOD].unique():
                    if DIMENSION in df.columns:
                        for dim in df[DIMENSION].unique():
                            row = {
                                DATASET: dataset,
                                MISSING_RATE: 0,
                                MISSINGNESS_TYPE: mt,
                                IMPUTATION_METHOD: imp,
                                DIMENSION: dim,
                            }
                            for metric in metrics:
                                row[metric] = 0.0
                            new_rows.append(row)
                    else:
                        row = {
                            DATASET: dataset,
                            MISSINGNESS_TYPE: mt,
                            IMPUTATION_METHOD: imp,
                        }
                        for metric in metrics:
                            row[metric] = 0.0
                        new_rows.append(row)

        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df.groupby(group)[metrics].mean().reset_index()

def boxplot_stats(df, group, metric, steps, ticks, filter=None):
    if filter is not None:
        df_stats = (
            df[filter]
            .groupby(group)[metric]
            .agg(['mean', 'sem'])
            .reset_index()
        )
    else:
        df_stats = (
            df
            .groupby(group)[metric]
            .agg(['mean', 'sem'])
            .reset_index()
        )
    return df_stats, np.arange(len(steps)), 0.8 / len(ticks)

def compute_mean_sem(type, data, group, metric=None, filter=None):
    if type == MISSING_RATE or type == MISSINGNESS_TYPE:
        if filter is not None:
            data = data[filter]
        return data.groupby(group)[metric].mean(), data.groupby(group)[metric].sem()
    elif type == IMPUTATION_METHOD or type == TDA_METHOD:
        return (data[filter]
                .set_index(group)
                .reindex(COLLECTIONS[group])[['mean', 'sem']]
                .to_numpy().T
            )

def plot_data(type, ax, mean, sem, label, color, x=None, width=None, linestyle='-'):
    if type == MISSING_RATE or type == MISSINGNESS_TYPE:
        ax.errorbar(
            mean.index, mean.values, yerr=sem.values, 
            label=label , marker='o', capsize=5, 
            color=color, linestyle=linestyle
        )
    elif type == IMPUTATION_METHOD or type == TDA_METHOD:
        ax.bar(x, mean, width, yerr=sem, capsize=5, label=label, color=color)

def format_axes(type, ax, x_label, y_label, legend=False, title=False, title_label=None, ticks=None, tick_labels=None):
    if type == MISSING_RATE or type == MISSINGNESS_TYPE:
        if legend:
            ax.legend(fontsize=12)
    elif type == IMPUTATION_METHOD or type == TDA_METHOD:
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=12)

    if title:
            ax.set_title(title_label, fontsize=16, weight='bold')
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True)

def legend(axes, fig, cols, bbox):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=cols,
        fontsize=12,
        bbox_to_anchor=bbox
    )

def multi_legend(fig):
    fig.legend(
        handles=[
            Line2D([0], [0], color=plt.get_cmap(COLOR_MAP[DIMENSION])(dim), lw=2) 
            for dim in DIMENSIONS
        ],
        labels=[f'{LABEL[DIMENSION]} {dim}' for dim in DIMENSIONS],
        loc='lower center',
        ncol=len(DIMENSIONS),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.04)
    )

    fig.legend(
        handles=[
            Line2D([0], [0], color='black', lw=2, linestyle=COLLECTIONS[LINESTYLE][j % len(COLLECTIONS[LINESTYLE])])
            for j, mt in enumerate(COLLECTIONS[MISSINGNESS_TYPE])
        ],
        labels=[LABEL[mt] for mt in COLLECTIONS[MISSINGNESS_TYPE]],
        loc='lower center',
        ncol=len(COLLECTIONS[MISSINGNESS_TYPE]),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.07)
    )

def plot(title=''):
    plt.title(title)
    plt.tight_layout(h_pad=2, w_pad=3)
    plt.show()
    plt.close()

def plot_missing_data(missing_data, original_data, type, title=''):
    """
    Visualize the correlation of missing data using a heatmap.

    :param missing_data: Data with missing values
    :param original_data: Original data without missing values
    """
    miss_plot = PlotMissingData(data_missing=missing_data, data_original=original_data)
    miss_plot.visualize_miss(type, save=False)
    plot(title)

def plot_persistence_diagram(data, title='', band=0.0, xlim=None, ylim=None, file=False):
    if file:
        ax = gudhi.plot_persistence_diagram(persistence_file=data, band=band)
    else:
        max_dim = int(max(item[2] for item in data[0]))
        persistence_intervals = [
            [item[0:2] for item in data[0] if item[2] == dim]
            for dim in range(max_dim + 1)
        ]
        ax = gudhi.plot_persistence_diagram(persistence_intervals, band=band)

    ax.set_title(title)
    ax.grid(True)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

def plot_barcode_diagram(data, title_0, title_1_2):
    H0 = [(int(row[0]), (row[1], row[2])) for row in data if int(row[0]) == 0]
    H1_H2 = [(int(row[0]), (row[1], row[2])) for row in data if int(row[0]) in [1, 2]]
    gudhi.plot_persistence_barcode(H0, legend=True)
    plt.title(title_0)
    plt.grid(True)
    gudhi.plot_persistence_barcode(H1_H2, legend=True)
    plt.ylim(len(H1_H2) + 2)
    plt.title(title_1_2)
    plt.grid(True)

def plot_persistence_landscape(diagrams, title=''):
    for k, l in enumerate(diagrams):
        plt.plot(l, color=['red', 'blue', 'green'][k], label=f'$\\lambda_{{{k+1}}}$')

    plt.title(title)
    plt.xlabel('$\\varepsilon$')
    plt.ylabel('$\\lambda_k(\\varepsilon)$')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_persistence_image(persistent_images, sample_idx = 0, title=''):
    n_dims = persistent_images.shape[1]
    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 4))

    for i in range(n_dims):
        ax = axes[i] if n_dims > 1 else axes
        ax.imshow(persistent_images[sample_idx, i])
        ax.set_title(f'H{i}')
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def persistence_diagram_missing_rate(dataset):
    dataset_mr = {}
    dataset_mr[0] = dataset[DATA]
    for mr in [5, 10, 25]:
        dataset_mr[mr] = IMPUTATION[KNN][FUNCTION](MISSINGNESS[MAR][FUNCTION](dataset[DATA], dataset[TARGET], mr, None), None)

    diagrams_mr = {}
    for mr, data in dataset_mr.items():
        diagrams_mr[mr] = normalize_by_diameter(TDA[VR][FUNCTION](data), dataset[DATA])

    births, deaths = [], []
    for diagrams in diagrams_mr.values():
        for pt in diagrams[0]:
            births.append(pt[0])
            deaths.append(pt[1])

    padding = 0.05 * (max(deaths) - min(births))
    xlim = (min(births) - padding, max(deaths) + padding)
    ylim = (min(births) - padding, max(deaths) + padding)
    for mr, diagrams in diagrams_mr.items():
        plot_persistence_diagram(diagrams, band=0.075, xlim=xlim, ylim=ylim)