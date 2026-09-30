# Packages
import numpy as np 
import matplotlib.pyplot as plt

# Import functions from shared model
from ecDNA_dynamics_model import evolve_population


# This .py file presents the code used to generate figure 1.3 for the results 'Reintegration truncates the ecDNA copy-number tail'. 


## --------------------------------------------------------- Result 1.3: Reintegration truncates the ecDNA copy-number tail ---------------------------------------------------------



# ----------------------------------------------------- FIGURE 1.3a -------------------------------------------------------
# Set parameters 
s = 1.5
N_target = 10**5
checkpoint_sizes = [1, 100, 500, 1000, 2000, 5000, 10000, 10**5]
r = 1.2
k = 0.1

# Control
population_0, next_slot_0, *_ = evolve_population(s, N_target, checkpoint_sizes, 0, r, k, p_excision)

# Reintegration (p_integrate = 0.01) 
population_re1, next_slot_re1, *_ = evolve_population(s, N_target, checkpoint_sizes, 0.01, r, k, p_excision)

# Reintegration (p_integrate = 0.05) 
population_re2, next_slot_re2, *_ = evolve_population(s, N_target, checkpoint_sizes, 0.05, r, k, p_excision)


# ------------- ecDNA copy-number distribution -------------

def distribution_Func(CN):
    # Keep only ecDNA+ cells 
    CN = CN[CN > 0]
    # Sort copy numbers into ascending order 
    CN = np.sort(CN)
    # Create a variable for denominator which is the count of ecDNA+ cells 
    n = len(CN) 
    # Rank-based count for a survival curve. Number of cells with copy number >= the i-th sorted value
    counts = np.arange(n, 0, -1) 
    # Create a fraction of cells greater or equal to each value for y-axis 
    distribution = counts/ n
    return CN, distribution  

# Plot ecDNA copy-number distribution
x_0, y_0 = distribution_Func(population_0[:next_slot_0, 0]) # p_integrate = 0
x_re1, y_re1 = distribution_Func(population_re1[:next_slot_re1, 0]) # p_integrate = 0.01
x_re2, y_re2 = distribution_Func(population_re2[:next_slot_re2, 0]) # p_integrate = 0.05
plt.plot(x_0, y_0, color="grey", linestyle="dashed", label="P=0")
plt.plot(x_re1, y_re1, color="blue", linestyle="dashed", label="P=0.01")
plt.plot(x_re2, y_re2, color="red", linestyle="dashed", label="P=0.05")

plt.rcParams.update({'font.size': 12})
plt.xscale("log")
plt.yscale("log")
plt.xlabel("ecDNA copy number")
plt.ylabel("Fraction of cells greater or equal to copy number")
plt.legend(loc="upper right")
plt.show()



# ----------------------------------------------------- FIGURE 1.3b -------------------------------------------------------
# This code chunk sweeps across the different reintegration probabilities (0.001, 0.005, 0.01, 0.02, 0.03, 0.05) holding all 
# the other parameters constant. For each probability, 50 ecDNA+ 'tumours' are grown to 10^5 cells, and the mean copy number is 
# recorded for each. These 50 per-tumour means are then averaged across the 50 runs and plotted against reintegration probability. 

#  set parameters 
s = 1.5
r = 1.2
k = 0.1
N_target = 10**5
N_runs = 50 
checkpoint_sizes = [100, 500, 1000, 2000, 5000, 10**4, 10**5] 
p_values = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]

# Initialise lists
avg_ecDNA_mean = []
ecDNA_spread = []
avg_HSR_mean = []
HSR_spread = []

# ------------- Mean copy ecDNA and HSR copy number across varying p_integration -------------

# Amplification means across different values of p-values 
for p in p_values:
    # Store the final means for each subtype/amplification in each run 
    final_ecDNA_mean = []
    final_HSR_mean = []

    # Run N-runs (50) independent simulations at this p-value
    for i in range(N_runs):
        # Call function and run one simulation at current p-value
        *_, all_ecDNA_mean, all_HSR_mean = evolve_population(s, N_target, checkpoint_sizes, p, r, k, p_excision)
        if len(all_ecDNA_mean) != len(checkpoint_sizes):
            continue
        # Store the final mean copy number in the final tumour
        final_ecDNA_mean.append(all_ecDNA_mean [-1])
        final_HSR_mean.append(all_HSR_mean [-1])

    # Mean of means for the final entry of each run
    avg_ecDNA_mean.append(np.mean(final_ecDNA_mean))
    avg_HSR_mean.append(np.mean(final_HSR_mean))
    # Standard deviation 
    ecDNA_spread.append(np.std(final_ecDNA_mean))
    HSR_spread.append(np.std(final_HSR_mean))

# Plot bar plot
x = np.arange(len(p_values))
width = 0.35

ecDNA_x = x - width/2 
HSR_x = x + width/2 

plt.bar(ecDNA_x, avg_ecDNA_mean, width, yerr = ecDNA_spread, color="orange", edgecolor="black", label="ecDNA")
plt.bar(HSR_x, avg_HSR_mean, width, yerr = HSR_spread, color="red", edgecolor="black", label="HSR")
plt.xticks(x, p_values)
plt.xlabel("reintegration probability (p)")
plt.ylabel("Mean copy number per cell")
plt.title("Mean copy number vs reintegration probability")
plt.legend()
plt.show()



# ----------------------------------------------------- FIGURE 1.3b -------------------------------------------------------
# This code sweeps across three distinct reintegration probabilities (p = 0.001 - 0.05)
# • p = 0.001 --> weak reintegration
# • p = 0.005 --> mild reintegration
# • p = 0.05 --> strong reintegration
# For each probability, 50 tumours are grown to 10^5 cells, and the ecDNA and HSR copy numbers from every cell across all
# 50 simulations are pooled together. The resulting distribution shows how many cells carry each copy number for each amplification type 
# across the pooled tumours

# Set parameters
s = 1.5
r = 1.2 
k = 0.1
N_runs = 50
N_target = 10**5
p_values = [0.001, 0.005, 0.05] 
checkpoint_sizes = [100, 500, 1000, 2000, 5000, 10**4]
ecDNA_pools = []
HSR_pools = []

# ------------- ecDNA and HSR copy number distributions across p_integration (0.001, 0.005, 0.05) -------------

for p in p_values: 
    ecDNA_copies = []
    HSR_copies = []
    for i in range(N_runs):
        # This grabs the arrays so we can access ecDNA and HSR copies, but this returns all copies
        population, next_slot, *_  = evolve_population(s, N_target, checkpoint_sizes, p, r, k, p_excision)
        # Therefore  filter through the arrays for ecDNA by using population[:next_slot, 0], and add it to the 
        # empty ecDNA_copies list 
        ecDNA_copies.append(population[:next_slot, 0])
        # The same for HSRs
        HSR_copies.append(population[:next_slot, 1])
    # Join all arrays from replicate runs to a single array 
    ecDNA_pools.append(np.concatenate(ecDNA_copies))
    HSR_pools.append(np.concatenate(HSR_copies))


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

# Loop over all the p_values for plotting 
for i in range(len(p_values)):
    ax = axes[i]
    p = p_values[i]
    ecDNA_pool = ecDNA_pools[i]
    HSR_pool = HSR_pools[i]

    # Set width of bars 
    bins = np.arange(0, max(ecDNA_pool.max(), HSR_pool.max()) + 2, 1)
    
    # Plot
    ax.hist(ecDNA_pool, bins=bins, alpha=0.7, density=True, color="orange", edgecolor="black", label="ecDNA+")
    ax.hist(HSR_pool, bins=bins, alpha=0.7, density=True, color="red", edgecolor="black", label="HSR")
    ax.set_xlabel("Copy number")
    ax.set_yscale("log")
    ax.set_title(f"p = {p}")

axes[0].set_ylabel("Frequency")    
axes[2].legend()
plt.tight_layout()
plt.show()


# Output: src/untreated_model/result_1.3_ecDNA_CN_truncation.png
