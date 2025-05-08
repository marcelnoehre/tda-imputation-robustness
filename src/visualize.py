import gudhi
from mdatagen.plots import PlotMissingData

def persistence_diagram(dataset):
    gudhi.plot_persistence_diagram(dataset)

def plot_missing_data(data, missing_data, style="normal"):
    miss_plot = PlotMissingData(data_missing=missing_data, data_original=data)
    miss_plot.visualize_miss(style)
