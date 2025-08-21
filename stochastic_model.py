import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate._ivp.ivp import OdeResult
from scipy.optimize import curve_fit
from utils import debug, from_molecules_to_uM, from_uM_to_molecules


def make_toggle_switch_stochastic(alpha_1, beta_1, alpha_2, beta_2,
                           K1, K2, d1_base, n, d2, gamma, epsilon, u_0_nmolecules, v_0_nmolecules, debug_flag, t_start_MMC=60, t_end_MMC=960, t_end_sim=1200):
    """
    Factory for build genetic switch stochastic (tau-leap) model
    """

    def d1(t, s):
        if t < t_start_MMC or t > t_end_MMC:
            return d1_base # outside the MMC induction window
        else:
            return d1_base + gamma * s / (1 + s) #during the MMC induction window
    
    def hill_function(x, K):
        """
        Function to be used in the ODEs to model the Hill function.
        Args:
            x : concentration of one of the two species (u or v)
            K : a parameter that sets the threshold for the Hill function

            Note: x and K should be in the same units 
        """
        return K**n / (K**n + x**n)

    def genetic_switch_stochastic(tau_initial, s, enable_bimodal_sampling=False):
        """
        One stochastic trajectory via Poisson tau-leap for the SOS-pathway genetic switch.
        
        Two strategy are implemented to avoid negative values for the number of molecules after the updates:
        1. Bimodal sampling: using binomial distribution for the degradation step
        2. Post-leap check: if the update leads to negative values, the step is discarded and tau is divided by 2.
        You can choose which one to use by setting the enable_bimodal_sampling flag.
        
        Returns times, u, v arrays.
        """
        ### convert the parameters from concentrations to molecular numbers
        alpha_1_nmol = 500 * alpha_1
        beta_1_nmol =  500 * beta_1
        alpha_2_nmol =  500 * alpha_2
        beta_2_nmol =  500 * beta_2
        K1_nmol  =  500 * K1
        K2_nmol =  500 * K2
        debug(f"Parameters converted:\n\talpha_1_nmol:{alpha_1_nmol}, beta_1_nmol: {beta_1_nmol},\n\talpha_2_nmol: {alpha_2_nmol}, beta_2_nmol: {beta_2_nmol},\n\tK1_nmol: {K1_nmol}, K2_nmol: {K2_nmol}\n", debug_flag)
        

        ### Initializiation 
        t = [0.0] 
        u = [u_0_nmolecules]
        v = [v_0_nmolecules]
        ##  Maybe update because of slowleness of lists
        debug(f"Step 0:\n sim time 0.0 min:\nInitial nr of molecules:\n\tu: {u[0]}, v: {v[0]} \n-----------------------\n", debug_flag)
        if u[0]<0 or v[0] < 0: raise Exception("Got negative value for inital molecular number: non valid")
 
        rng = np.random.default_rng() #generatore di numeri casuali


        ### Tau-leap algorithm with post-leap check
        k=0 # simulation step index
        current_tau = tau_initial
        tk = t[0]
        while tk < t_end_sim and current_tau > 0.01: # if t reaches the t_end_MMC or tau reaches the lower bound, exit the loop

            uk, vk = u[-1], v[-1] #previous values (needed to compute updates)

            ### propensity rates (molecules/min)
            prop_u = epsilon * (alpha_1_nmol + beta_1_nmol * hill_function(vk, K1_nmol))
            prop_v = epsilon * (alpha_2_nmol + beta_2_nmol * hill_function(uk, K2_nmol))
            debug(f"prop_u: {prop_u}, prop_v: {prop_v}",  debug_flag)
            if prop_u<0 or prop_v < 0: raise Exception("Got negative value for propensity functions: non valid as the Poisson mean must be positive")#ensure that the mean is not negative: poisson property

            ### degradation rates (molecules/min)
            deg_u = d1(tk, s) * uk #ensure number of molecules is not negative, otherwise subtracting deg_ in the updates will lead to an increase 
            deg_v = d2 * vk 
            debug(f"deg_u: {deg_u}, deg_v: {deg_v}",  debug_flag)
            if deg_u<0 or deg_v < 0: raise Exception("Got negative value for propensity functions: non valid as the Poisson mean must be positive")#ensure that the mean is not negative: poisson property


            ### Poisson jumps with mean = rate*tau for the increase
            #sampling number of molecules produced
            dU_increase = rng.poisson(prop_u * current_tau) 
            dV_increase = rng.poisson(prop_v * current_tau)
            debug(f"dU_increase: {dU_increase}, dV_increase: {dV_increase}",  debug_flag)

            # sampling number of molecules degradated 
            if enable_bimodal_sampling: #binomial strategy
                print("TODO")
            #     # Binomial jumps (degradation) starting from total propensities deg_u = d1*uk, deg_v = d2*vk
            #     if uk > 0:
            #         p_u = 1.0 - np.exp(-(deg_u / uk) * tau)   # per-molecule probability (GPT)
            #         dU_decrease = rng.binomial(int(uk), p_u)
            #     else:
            #         dU_decrease = 0

            #     if vk > 0:
            #         p_v = 1.0 - np.exp(-(deg_v / vk) * tau) # per-molecule probability (GPT)
            #         dV_decrease = rng.binomial(int(vk), p_v)
            #     else:
            #         dV_decrease = 0
            
            else: #post leap check strategy
                dU_decrease = rng.poisson(deg_u * current_tau)
                dV_decrease = rng.poisson(deg_v * current_tau)
            debug(f"dU_decrease: {dU_decrease}, dV_decrease: {dV_decrease}",  debug_flag)


            ### updates
            uk = uk + dU_increase - dU_decrease
            vk = vk + dV_increase - dV_decrease

            #Post leap check:
            if uk < 0 or vk < 0:
                debug("**************************************\nGot negative values for the updates, discarding the updates and repeating the step after dividing tau by half", True) 
                current_tau /= 2
                debug(f"Updated tau: {current_tau} min\n**************************************", True)        
        
            else: 
                tk += current_tau #update time: this is the current time of simulation

                k += 1 #update step

                #storing values
                t.append(tk) # current time of sim
                u.append(uk)
                v.append(vk)
                
                debug(f"Step {k+1}:\t\t sim time: {tk+current_tau} min \t\tu: {uk}, v: {vk}", debug_flag)

        if t[-1] < t_end_MMC: debug(f"Simulation failed because of threshold for tau reached.\n Time achieved: {t[-1]}")
        else: debug("Successful simulation", debug_flag)

        # Convert lists to numpy arrays for consistency
        t = np.array(t)
        u = np.array(u)
        v = np.array(v)

        return t, u, v
    
    return genetic_switch_stochastic


def check_if_switched(sim, molecules_per_uM, debug_flag=False):
    """
    This function checks if the system has switched.
    It returns True if the system has switched, False otherwise.

    Initially we have 2125 molecules of λCI (u) and 125 molecules of LacR (v).
    The system is considered to have switched if,  after a certain time, the number of molecules of λCI (v) is greater than 1700 and the number of molecules of LacR (u) is less than 300 (on average).
    
    The function deals with either deterministic or stochastic results. 
    When the results are deterministic, it converts the concentrations from uM to number of molecules using the molecules_per_uM conversion factor.
    When the results are stochastic, it assumes that the number of molecules is already in the correct format.

    Args:
        sim: the simulation result, either an OdeResult object or a tuple (time, u, v)
        molecules_per_uM: conversion factor from uM to number of molecules
    Returns:
        switched: True if the system has switched, False otherwise
        v_1140: number of molecules of LacR at t=1140 min (for Fig3 histogram)
    """
    if isinstance(sim, OdeResult):  #deterministic results
        t, u, v = sim.t, sim.y[0], sim.y[1]
        #convert to number of molecules
        u = from_uM_to_molecules(u, molecules_per_uM)
        v = from_uM_to_molecules(v, molecules_per_uM)
    
    else: #stochastic results
        t, u, v = sim[0], sim[1], sim[2]

    #check if the system has switched
    index_time1000_min = np.where(t >= 1000)[0]
    if index_time1000_min.size == 0:
        raise Exception("No data for t >= 1000 min")

    lastN = 200 # number of samples to check for the switch
    if index_time1000_min.size < lastN:
        raise Exception(f"Not enough data points after 1000 min to check for the switch. Found {index_time1000_min.size} points, but need at least {lastN} points.")
    
    # Get the last N samples after 1000 min
    u_lastN = u[index_time1000_min[-lastN:]]
    v_lastN = v[index_time1000_min[-lastN:]]

    # Calculate the mean of the last N samples
    # Note: the lastN samples are taken from the end of the simulation, not from the end of the MMC window
    u_lastN_mean = np.mean(u_lastN)
    v_lastN_mean = np.mean(v_lastN)

    switched = (u_lastN_mean < 300) and (v_lastN_mean > 1700) # check the last 100 values to see if the system has switched
    
    if switched:
        debug(f"The system has switched:", debug_flag)
    else:
        debug(f"The system has NOT switched:", debug_flag)
    debug(f"\tlast {lastN} samples averages:\n\t\tu={round(u_lastN_mean,1)},\n\t\tv={round(v_lastN_mean,1)}", debug_flag)
    
    # check the LacR number of molecules at t=1140 (for Fig3 histogram)
    if t[-1] < 1140:
        raise Exception(f"Simulation ended before t=1140 min. Last time point: {t[-1]}. Cannot check the number of molecules at t=1140 min.")
    else:
        index_time1140_min = np.where(t >= 1140)[0][0]
        v_1140 = v[index_time1140_min]


    return switched, v_1140


def compute_percentage_switched(stochastic_model, tau, N_sims = 1000, molecules_per_uM=500):
    """
    This function computes the percentage of switched systems for different s values.
    It runs the stochastic model for each s value and checks if the system has switched.
    It returns a list of percentages of switched systems for each s value.
    The function also stores the results in a CSV file for further analysis.
    Args:
        stochastic_model: the stochastic model function
        tau: the time step for the simulation
        N_sims: number of simulations to run for each s value
        molecules_per_uM: conversion factor from uM to number of molecules
    """

    s_values = np.linspace(1.5,1.85, 15)

    percentages_switched_systems = []
    for s in s_values:
        print(f"Running simulations for s={s}")
        sims = [stochastic_model(tau, s) for _ in range(N_sims)]
        switched_systems = sum([check_if_switched(sim, molecules_per_uM)[0] for sim in sims]) 
        
        percentage_switched = (switched_systems / N_sims) * 100
        percentages_switched_systems.append(percentage_switched)
        print(f"\tPercentage of switched systems for s={s}: {percentage_switched:.2f}%")
        print()

    #Store results in a file
    results_df = pd.DataFrame({
        's': s_values,
        'percentage_switched': percentages_switched_systems
    })
    results_df.to_csv('results/fig2d_results.csv', index=False)

    return results_df


def fit_and_plot_hill_function(data_path='results/fig2d_results.csv', n = 4):
    """
    Fit a Hill function to the data of percentage of switched systems vs s value.
    This function retrieves data from a CSV file, fits a Hill function to the observed data,
    rescales the fit to match a specific point, and plots the observed data along with the fitted Hill function.

    Args:
        data_path: path to the CSV file containing the data
        n: Hill coefficient (fixed to 4 as per the paper)

    """
    #### RETRIEVING DATA ####

    # Load the data from the CSV file
    data = pd.read_csv(data_path)
    s_values = data['s'].values
    percentages_switched_systems = data['percentage_switched'].values

    # Convert data to a numpy array for curve fitting
    s = np.asarray(s_values, dtype=float)
    p_obs = np.asarray(percentages_switched_systems, dtype=float)


    #### FITTING DATA ####
    

    def hill_function(s, A, B, s0):
        """
        Base Hill function to fit the data, and the plot. A small value is added to the denominator to avoid division by zero.
        """
        x = s - s0   

        return 100.0 * A * (x**n) / (B**n + x**n + 1e-8)

    # initial guesses 
    A_guess = 1.0
    B_guess  = 0.25 #from paper
    s0_guess = 1.5 # because it is the first point in the s_values
    p0  = [A_guess, B_guess, s0_guess]


    # bounds: B>0; s<1.5, A>0
    bounds = ([1e-9, 1e-9, 1e-9], [10, 1e3, s.min()]) #([lower bounds], [upper bounds])

    pars, cov = curve_fit(hill_function, s, p_obs, p0=p0, bounds=bounds)
    A_fit, B_fit, s0_fit = pars
    print(f"Fit base: A={A_fit:.4f}, B={B_fit:.4f}, s0={s0_fit:.4f}, n={n}")


    #### RESCALING FIT FUNCTION ####

    # compute A to match the last point simulated 
    # retrieve the last point
    idx_ref = int(np.argmax(p_obs))
    s_ref = float(s[idx_ref])
    p_target = float(p_obs[idx_ref])    

    # Rescale A to match the target percentage at s_ref
    x_ref = max(s_ref - s0_fit, 0.0)
    hill_base_at_ref = 100.0 * (x_ref**n) / (B_fit**n + x_ref**n + 1e-12)
    A_rescale = p_target / (hill_base_at_ref + 1e-12)

    print(f"Scaled A to hit p(s_ref={s_ref:.3f})={p_target:.2f}% -> A={A_fit:.4f}")


    plt.figure(figsize=(10, 6))

    plt.scatter(s, p_obs, marker='o', color='red', facecolors='none', label='Simulated Data')

    s_grid = np.linspace(s.min(), s.max(), 400)
    plt.plot(s_grid, hill_function(s_grid, A_rescale, B_fit, s0_fit),
            label=r'Hill function   $100.0\cdot\frac{ A \cdot (s-s_0)}{B^n + (s-s_0)^n}$' + f'  (A={A_fit:.4f}, B={B_fit:.3f}, s0={s0_fit:.3f}, n={n})',
            linestyle='--', color='black')

    plt.xlabel('s value')
    plt.ylabel('Percentage of switched systems (%)')
    plt.title('Percentage of switched systems vs s value')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('results/fig2d_results.svg', format='svg')
    plt.show()



