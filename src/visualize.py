import gudhi
import matplotlib.pyplot as plt
from src.constants import MISSING_RATES

def persistence_diagram(dataset):
    gudhi.plot_persistence_diagram(dataset)

def plot_distance_over_missingrates(distances, distance_type):
    plt.figure(figsize=(8,5))
    plt.plot(MISSING_RATES, distances[0], marker='o', color='blue', label='H0')
    plt.plot(MISSING_RATES, distances[1], marker='o', color='orange', label='H1')
    plt.plot(MISSING_RATES, distances[2], marker='o', color='green', label='H2')

    plt.xlabel('Missing Rate')
    plt.ylabel(f'{distance_type} Distance')
    plt.title(f'{distance_type} Distance over Missing Rates by Dimension')
    plt.legend()
    plt.grid(True)
    plt.show()