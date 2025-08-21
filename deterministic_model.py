from scipy.integrate import solve_ivp

def make_toggle_switch_ode(alpha_1, beta_1, alpha_2, beta_2,
                           K1, K2, d1_base, n, d2, gamma, epsilon, t_start_MMC=60, t_end_MMC=960):
    """
    Factory for build genetic switch deterministic  model

    Here we use concentrations in uM for the variables
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

    def genetic_switch_ode(t, y, s):
        u, v = y # current values. u= λCI, v = LacR
        du = ( epsilon * (alpha_1 + beta_1 * hill_function(v, K1)) ) - d1(t, s) * u
        dv = ( epsilon * (alpha_2 + beta_2 * hill_function(u, K2)) ) - d2 * v
        return [du, dv]

    return genetic_switch_ode

def run_sim(s, ode_system, y0, t_sim, delta_t):
    """
    Run a simulation of the genetic switch model using the provided ODE system.
    This function uses the `solve_ivp` function from `scipy.integrate` to solve the ODEs.
    It returns the solution object which contains the time points and the corresponding values of u and v.
    """
    sol = solve_ivp(lambda t, y: ode_system(t, y, s),
                    t_sim, y0, max_step=delta_t)
    return sol