import matplotlib.pyplot as plt
from scipy.integrate._ivp.ivp import OdeResult

from stochastic_model import check_if_switched



def plot_trajectories(s_values, sims, t_start_MMC=60, t_end_MMC=960, filename="fig2.svg", save_plot=True):
    """
    This function plots the evolution of the system for different s values.
    It can be used both for the deterministic system, where data is measured in uM,
    and for the stochastic system, where data is measured in number of molecules.

    Args:
        s_values: list of s values for which the simulations were run
        sims: list of simulation results, where each result corresponds to a value in s_values

        Note:
        The type of elements in sims can be either:
            - OdeResult: for deterministic results, where the solution is an object with attributes t and y
            - tuple: for stochastic results, where the first element is time, the second is the number of molecules of u, 
                     and the third is the number of molecules of v 
    """
    num_curves = len(s_values) 
    if num_curves>5: raise Exception("Too many curves. Max number with current setting is 6. If you need more, change the color list")
    # Update: safety check: number of trajectories coincide with number of s values

    # Plot curves
    plt.figure(figsize=(8,5))

    color_idx, color_list = 0, list(plt.colormaps['Paired'].colors)
    for s, sol in zip(s_values, sims):
        
        #load the solutions
        if isinstance(sol, OdeResult):  #deterministic results
            t, u, v = sol.t, sol.y[0], sol.y[1]
            ylabel = "concentrations (μM)"
            sim_type= "Deterministic"
        else: #stochastic results
            t, u, v = sol
            ylabel = "number of molecules"
            sim_type= "Stochastic"

        #plot the curves
        if num_curves > 1:
            plt.plot(t, u, color=color_list[color_idx],  label=f"s={s}, LacR ", linewidth=2, alpha=0.7)
            plt.plot(t, v, color=color_list[color_idx+1], label=f"s={s}, λCI ", linewidth=2, alpha=0.7)
            title = sim_type + f" genetic switch model for multiple s values"
            
        else:
           plt.plot(t, u, color="red",  label=f"LacR", linewidth=0.7, alpha=0.8)
           plt.plot(t, v, color="blue", label=f"λCI", linewidth=0.7, alpha=0.8)
           title = sim_type + f" genetic switch model (s={s})"

        color_idx+=2

    plt.ylabel(ylabel)
    plt.xlabel("time (min)")
    plt.axvspan(t_start_MMC, t_end_MMC, alpha=0.15, color='gray', label='MMC induction window')
    plt.legend()
    plt.tight_layout()
    plt.title(title)
    if save_plot: plt.savefig(filename, format='svg')
    plt.show()
    

def plot_histograms_LacR(stochastic_model, molecules_per_uM=500):    
    """
    This function plots histograms of the number of LacR molecules after a certain time for different s values.
    It runs the stochastic model for each s value and checks if the system has switched.
    It then plots the histograms of the number of LacR molecules for each s value.
    Args:
        stochastic_model: the stochastic model function
        molecules_per_uM: conversion factor from uM to number of molecules
    """

    s_values = [1.3, 1.7, 1.75, 2.0]
    tau = 0.5

    N_sims = 1000 # number of simulations to run for each s value

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid
    axes = axes.flatten()  # flatten into 1D array for easy indexing

    for i, s in enumerate(s_values):
        print(f"Running simulations for s={s}")
        sims = [stochastic_model(tau, s) for _ in range(N_sims)]
        LacR_values = [check_if_switched(sim, molecules_per_uM)[1] for sim in sims]

        binwidth = 3
        axes[i].hist(
            LacR_values,
            bins=range(0, 2200 + binwidth, binwidth),
            color='black'
        )
        axes[i].set_xlabel("Number of LacR molecules")
        axes[i].set_ylabel("Number of cells")
        axes[i].set_xlim(0, 2200)
        axes[i].set_title(f"s={s}")

    plt.tight_layout()
    plt.savefig("results/fig3.svg", format="svg")  # save to SVG
    plt.show()