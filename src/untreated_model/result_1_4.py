# Packages
import numpy as np 
import matplotlib.pyplot as plt
rng = np.random.default_rng()

# Import functions from shared model
from ecDNA_dynamics_model import evolve_population

# This .py file presents the code used to generate figure 1.4 for the results 'Reintegration and excision emergence appears sequentially'. 

## --------------------------------------------------------- Result 1.3: Reintegration and excision emergence appears sequentially ---------------------------------------------------------

# Set parameters
s = 1.5
r = 1.2 
k = 0.1
N_runs = 50
N_target = 10**5
p_integrate = 0.01
p_excision = 0.02
excision_pool = []
reintegration_pool = []
checkpoint_sizes = [100, 500, 1000, 2000, 5000, 10**4]

# Over the 50 runs, record an excision or reintegration event occurring. 
# In each run the events will be appended to the respective lists 
for i in range(N_runs):
    *_, event_excision, event_reintegration = evolve_population(s, N_target, checkpoint_sizes, p_integrate, r, k, p_excision)
    excision_pool.append(event_excision)
    reintegration_pool.append(event_reintegration)
  

# Append all 50 lists together
excision_times = np.concatenate(excision_pool) 
reintegration_times = np.concatenate(reintegration_pool)


# Use np.linspace instead np.arange to change the axis from integers to continuous decimals (time = generations) 
bins = np.linspace(0, 17, 30) 


# Plot 
plt.hist(reintegration_times, bins=bins, alpha=0.7, color="darkseagreen", edgecolor="black", label="Reintegration")
plt.hist(excision_times, bins=bins, alpha=0.5, color="plum", edgecolor="black", label="Excision")
plt.xlabel("Time (Generations)")
plt.ylabel("Frequency")
plt.yscale("log")
plt.legend()
plt.show()
