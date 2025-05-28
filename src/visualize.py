import gudhi
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from src.constants import MISSING_RATES, DIM_COLORS

def plot_persistence_diagram(diagrams, title=''):
    max_dim = int(max(item[2] for item in diagrams[0]))
    persistence_intervals = [
        [item[0:2] for item in diagrams[0] if item[2] == dim]
        for dim in range(max_dim + 1)
    ]
    ax = gudhi.plot_persistence_diagram(persistence_intervals)
    ax.set_title(title)

def plot_persistence_landscape(diagrams, title=''):
    for k, l in enumerate(diagrams):
        plt.plot(l, color=DIM_COLORS[k], label=f"$\\lambda_{{{k+1}}}$")

    plt.title(title)
    plt.xlabel("$\\varepsilon$")
    plt.ylabel("$\\lambda_k(\\varepsilon)$")
    plt.legend()
    plt.grid(True)
    plt.show()


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