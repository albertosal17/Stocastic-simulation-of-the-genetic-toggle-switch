import numpy as np
import pandas as pd
from tqdm import tqdm

from stochastic_model import check_if_switched
from utils import debug

def compute_binarized_matrix(stochastic_model, s, tau_initial, debug_mode, n_simulations=28000, molecules_per_uM=500):
    """
    This function runs the stochastic model for a specific value of s, multiple times. For
    each simulation, it checks which of the two species is present in a higher concentration:
    it assigns 1 to the species that is more present, and 0 to the other. Storing the results of a 
    lot of simulation, it allows to re-create the binarized matrix needed for the quantum method 
    for reconstruction of the gene regulatory network.
    """

    binarized_matrix = np.zeros((2, n_simulations), dtype=int)
    for n_sim in tqdm(range(n_simulations)):
        sol = stochastic_model(tau_initial, s)
        switched = check_if_switched(sol, molecules_per_uM) #true if lambaCI > LacR, false otherwise
            
        binarized_matrix[0, n_sim] = int(switched[0]) # lambdaCI (0 index)
        binarized_matrix[1, n_sim] = int(not switched[0]) # LacR (1 index)
    
    # Construct a pandas dataframe: rows names are the species names, and columns are the simulation results (cell number)
    binarized_df = pd.DataFrame(binarized_matrix, index=['λCI', 'LacR'])
    # Store the binarized matrix in a csv file
    binarized_df.to_csv(f'quantum_results/binarized_matrix_s_{str(s).replace('.', '_')}_nsim_{n_simulations}.csv', index=True, header=True)


    return binarized_df, s