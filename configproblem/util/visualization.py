import math
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from configproblem.util.hamiltonian_math import compute_config_energy
from configproblem.qaoa_mincost_sat import get_expectation_statevector


def plot_beta_gamma_cost_landscape(hamiltonians, strategies, nqubits, step_size):
    """
        Plots the cost landscape for different values of beta and gamma
        for a given list of hamiltonians and list of strategies.

        :param hamiltonians: list of hamiltonians to plot
        :param strategies: list of strategies to plot
        :param nqubits: number of qubits
        :param step_size: step size for beta and gamma, value will be doubled for gamma as it's limits are also doubled
    """
    plot_arguments = []
    for hamiltonian in hamiltonians:
        for strategy in strategies:
            plot_arguments.append({"hamiltonian": hamiltonian, "strategy": strategy})

    x_axis = np.arange(-math.pi, math.pi, step_size)
    y_axis = np.arange(-2 * math.pi, 2 * math.pi, 2 * step_size)

    fig, axes = plt.subplots(len(hamiltonians), len(strategies), figsize=(18, 16))
    fig.suptitle(r"cost landscape for different values of $\gamma$ and $\beta$", fontsize=20)
    fig.tight_layout(pad=5.0)
    cmap = "viridis"

    for ax, arguments in zip(axes.ravel(), plot_arguments):
        ax.set_title(f"Hamiltonian: {arguments['hamiltonian']['name']}\n strategy: {arguments['strategy']}")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\gamma$")

        hamiltonian = arguments["hamiltonian"]["hamiltonian"]

        expectation = np.zeros(shape=(len(x_axis), len(y_axis)))
        expectation_max = -sys.maxsize - 1
        expectation_min = sys.maxsize

        for i_index, i in enumerate(x_axis):
            for j_index, j in enumerate(y_axis):
                expectation_function = get_expectation_statevector(hamiltonian, nqubits, 1)
                value = expectation_function({"beta": i, "gamma": j})
                expectation[i_index][j_index] = value
    
                if expectation_max < value:
                    expectation_max = value
                elif expectation_min > value:
                    expectation_min = value

        pcm = ax.pcolormesh(x_axis, y_axis, expectation, shading="gouraud", cmap=cmap, vmin=expectation_min,
                            vmax=expectation_max)
        plt.colorbar(pcm, ax=ax)

    plt.show()


def plot_f_mu_cost_landscape(hamiltonian, nqubits):
    """
        Plots the cost landscape for f and mu for a given hamiltonian where f is the function for the config energy
        and mu is the average difference in energy between the current config and all configs with hamming distance 1.

        :param hamiltonian: hamiltonian used to calculate f
        :param nqubits: number of qubits
    """
    # Calculate f(z) using the given hamiltonian for each bitstring z
    f = np.zeros(shape=(2 ** nqubits))

    for i in range(0, 2 ** nqubits):
        bitstring = np.binary_repr(i, width=nqubits)
        config = [-1 if s == "0" else 1 for s in bitstring]
        f[i] = compute_config_energy(hamiltonian, config)

    # Calculate mu(z_0) = sum_{\Delta(z_0, z) = 1} \frac{f(z) - f(z_0)}{N} for each bitstring z_0
    mu = np.zeros(shape=(2 ** nqubits))

    for i in range(0, 2 ** nqubits):
        bitstring = np.binary_repr(i, width=nqubits)
        for j in range(0, 2 ** nqubits):
            bitstring_j = np.binary_repr(j, width=nqubits)
            hamming_distance = 0
            for char_i, char_j in zip(bitstring, bitstring_j):
                if char_i != char_j:
                    hamming_distance += 1
            if hamming_distance == 1:
                mu[i] += (f[j] - f[i]) / nqubits

    f_mu = np.vstack([mu, f])
    z = gaussian_kde(f_mu)(f_mu)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

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

    plt.show()


def plot_counts_histogram(counts, best_config, valid_configs):
    """
        Plots a histogram of the counts for each possible config.
        The best config is highlighted in red and valid configs are highlighted in brown.

        :param counts: dictionary containing the counts for each config that was measured
        :param best_config: valid config with the lowest energy
        :param valid_configs: list of valid configs
    """
    for i in range(0, 2 ** 6):
        if not counts.keys().__contains__(np.binary_repr(i, 6)):
            counts[np.binary_repr(i, 6)] = 0

    # Sort histogram
    sorted_keys = sorted(counts.keys())
    sorted_items = sorted(counts.items(), key=lambda item: item[0])
    sorted_values = [item[1] for item in sorted_items]

    col = []
    for key in sorted_keys:
        if key == best_config:
            col.append('r')
        elif valid_configs.__contains__(key):
            col.append("brown")
        else:
            col.append('b')

    plt.rcParams["figure.figsize"] = (40, 10)
    plt.ylabel("Count")
    plt.xticks(rotation=70, ha="right")
    plt.bar(sorted_keys, sorted_values, width=0.5, color=col)

    max_count = 0
    for val in sorted_values:
        if val > max_count:
            max_count = val

    for i, val in enumerate(sorted_values):
        plt.text(x=i, y=val + max_count / 100, s=f"{val}", fontdict=dict(fontsize=10), horizontalalignment="center")

    plt.axhline(y=counts[best_config], color='r', linestyle='--')

    red_patch = mpatches.Patch(color='r', label="best config")
    brown_patch = mpatches.Patch(color="brown", label="valid config")
    blue_patch = mpatches.Patch(color='b', label="invalid config")
    plt.legend(handles=[red_patch, brown_patch, blue_patch], loc="upper right")

    plt.show()
