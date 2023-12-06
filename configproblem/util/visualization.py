import math
import sys
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qiskit.result import Counts
from qubovert.utils import DictArithmetic
from scipy.stats import gaussian_kde
from configproblem.util.hamiltonian_math import compute_config_energy
from configproblem.qaoa.qaoa_application import get_expectation_statevector


def plot_beta_gamma_cost_landscape(problem_circuit: Callable, hamiltonians: list[dict], strategies: list[str],
                                   nqubits: int, step_size: float, plot_titles: bool = True, file_path: str = None):
    """
        Plots the cost landscape for different values of beta and gamma
        for a given list of hamiltonians and list of strategies.

        :param problem_circuit: The function for creating the corresponding problem quantum circuit
        :param hamiltonians: list of hamiltonians to plot
        :param strategies: list of strategies to plot
        :param nqubits: number of qubits
        :param step_size: step size for beta and gamma, value will be doubled for gamma as it's limits are also doubled
        :param plot_titles: whether to plot titles for each subplot and a title for the whole figure
        :param file_path: path to save the figure to
    """
    plot_arguments = []
    for hamiltonian in hamiltonians:
        for strategy in strategies:
            plot_arguments.append({"hamiltonian": hamiltonian, "strategy": strategy})

    x_axis = np.arange(-math.pi, math.pi, step_size)
    y_axis = np.arange(-2 * math.pi, 2 * math.pi, 2 * step_size)

    fig, axes = plt.subplots(len(hamiltonians), len(strategies), figsize=(6 * len(strategies), 5.5 * len(hamiltonians)))
    if plot_titles:
        fig.suptitle(r"cost landscape for different values of $\gamma$ and $\beta$", fontsize="xx-large")
    fig.tight_layout(pad=5.0)
    cmap = "viridis"

    if len(hamiltonians) == 1 and len(strategies) == 1:
        axes_and_arguments = zip([axes], plot_arguments)
    else:
        axes_and_arguments = zip(axes.flat, plot_arguments)

    for ax, arguments in axes_and_arguments:
        if plot_titles:
            ax.set_title(f"Hamiltonian: {arguments['hamiltonian']['name']}\n Strategy: {arguments['strategy']}")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\gamma$")

        hamiltonian = arguments["hamiltonian"]["hamiltonian"]
        strategy = arguments["strategy"]

        expectation = np.zeros(shape=(len(x_axis), len(y_axis)))
        expectation_max = -sys.maxsize - 1
        expectation_min = sys.maxsize

        for i_index, i in enumerate(x_axis):
            for j_index, j in enumerate(y_axis):
                expectation_function = get_expectation_statevector(problem_circuit, hamiltonian, nqubits, 1,
                                                                   strategy=strategy)
                value = expectation_function([i, j])
                expectation[i_index][j_index] = value

                if expectation_max < value:
                    expectation_max = value
                elif expectation_min > value:
                    expectation_min = value

        pcm = ax.pcolormesh(x_axis, y_axis, expectation, shading="gouraud", cmap=cmap, vmin=expectation_min,
                            vmax=expectation_max)
        plt.colorbar(pcm, ax=ax)

    if file_path:
        plt.savefig(file_path, dpi='figure', transparent=True)

    plt.show()


def plot_f_mu_cost_landscape(hamiltonian: DictArithmetic, nqubits: int, plot_title: bool = True, file_path: str = None):
    """
        Plots the cost landscape for f and mu for a given hamiltonian where f is the function for the config energy
        and mu is the average difference in energy between the current config and all configs with hamming distance 1.

        :param hamiltonian: hamiltonian used to calculate f
        :param nqubits: number of qubits
        :param plot_title: whether to plot a title for the figure
        :param file_path: path to save the figure to
    """
    configurations = np.arange(2 ** nqubits)

    # Calculate f(z) using the given hamiltonian for each bitstring z
    ising_configs = np.array(((configurations[:, None] & (1 << np.arange(nqubits))) > 0)) * 2 - 1
    f = np.array([compute_config_energy(hamiltonian, bitstring) for bitstring in ising_configs])

    # Calculate mu(z_0) for each bitstring z_0
    neighbors = np.bitwise_xor(configurations[:, None], 1 << np.arange(nqubits))
    mu = np.sum((f[neighbors] - f[configurations][:, None]) / nqubits, axis=1)

    f_mu = np.vstack([mu, f])
    z = gaussian_kde(f_mu)(f_mu)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    if plot_title:
        fig.suptitle(r"cost landscape for $f$ and $\mu$", fontsize=20)

    ax.scatter(mu, f, c=z, s=40 / nqubits)
    ax.set_xlabel(r"$\mu$", fontsize=15)
    ax.set_ylabel(r"$f$", fontsize=15)
    ax.set_xlim(min(mu), max(mu))
    ax.set_ylim(min(f), max(f))

    divider = make_axes_locatable(ax)

    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    ax_histx.hist(mu)
    ax_histy.hist(f, orientation='horizontal')

    if file_path:
        plt.savefig(file_path, dpi='figure', transparent=True)

    plt.show()


def plot_counts_histogram(counts: Counts, nqubits: int, best_config: str, valid_configs: list[str],
                          file_path: str = None):
    """
        Plots a histogram of the counts for each possible config.
        The best config is highlighted in red and valid configs are highlighted in brown.

        :param counts: dictionary containing the counts for each config that was measured
        :param nqubits: number of qubits
        :param best_config: valid config with the lowest energy
        :param valid_configs: list of valid configs
        :param file_path: path to save the figure to
    """
    for i in range(0, 2 ** nqubits):
        if not counts.keys().__contains__(np.binary_repr(i, nqubits)):
            counts[np.binary_repr(i, nqubits)] = 0

    counts = {"".join(reversed(key)): value for key, value in counts.items()}
    valid_configs = ["".join(reversed(config)) for config in valid_configs]
    best_config = "".join(reversed(best_config))

    # Sort the counts dictionary by key
    sorted_counts = dict(sorted(counts.items()))

    col = []
    for key in sorted_counts.keys():
        if key == best_config:
            col.append('r')
        elif valid_configs.__contains__(key):
            col.append("brown")
        else:
            col.append('b')

    plt.rcParams["figure.figsize"] = (40, 10)
    plt.ylabel("Count")
    plt.xticks(rotation=70, ha="right")
    plt.bar(sorted_counts.keys(), sorted_counts.values(), width=0.5, color=col)

    max_count = 0
    for val in sorted_counts.values():
        if val > max_count:
            max_count = val

    for i, val in enumerate(sorted_counts.values()):
        plt.text(x=i, y=val + max_count / 100, s=f"{val}", fontdict=dict(fontsize=10), horizontalalignment="center")

    if sorted_counts.keys().__contains__(best_config):
        plt.axhline(y=counts[best_config], color='r', linestyle='--')

    red_patch = mpatches.Patch(color='r', label="best config")
    brown_patch = mpatches.Patch(color="brown", label="valid config")
    blue_patch = mpatches.Patch(color='b', label="invalid config")
    plt.legend(handles=[red_patch, brown_patch, blue_patch], loc="upper right")

    if file_path:
        plt.savefig(file_path, dpi='figure', transparent=True)

    plt.show()
