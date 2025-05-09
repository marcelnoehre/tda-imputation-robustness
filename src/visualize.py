import gudhi
import matplotlib.pyplot as plt
from mdatagen.plots import PlotMissingData

def persistence_diagram(dataset):
    gudhi.plot_persistence_diagram(dataset)

def plot_missing_data(data, missing_data, style="normal"):
    miss_plot = PlotMissingData(data_missing=missing_data, data_original=data)
    miss_plot.visualize_miss(style)

def plot_bar_chart(labels, values, x, y, title):
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values)
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, str(round(v, 3)), ha='center', va='bottom', fontsize=10)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.ylim(0, max(values) + 1)
    plt.show()
