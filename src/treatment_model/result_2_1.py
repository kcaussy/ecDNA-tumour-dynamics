
# This .py file presents the code used to generate figure 2.1 for the results 'ecDNA copy-number tail is truncated by treatment and partially rebuilt by growth'. 

# Packages
import numpy as np 
import matplotlib.pyplot as plt
rng = np.random.default_rng()

# Import functions from shared model
from ecDNA_dynamics_model import evolve_population

# Import functions from treatment model
from treatment_model import evolve_population_after_treatment

# Set parameters
s = 1.5
r = 1.2
k = 0.1
N_runs = 50
N_target = 10**5
p_integrate = 0.01
p_excision = 0.02
checkpoint_sizes = [100, 500, 1000, 2000, 5000, 10**4]

# Iterate loop 50 times:
# Number of cells pre-treatment
ePos_pre_treatment = []
eNeg_pre_treatment = [] 
HSR_pre_treatment = []

# Max copies for ecDNA and HSRs in pre-treatment:
max_ecDNA_pre_treatment = []   
max_HSR_pre_treatment = []

# Max copies for ecDNA and HSRs right after treatment (killing of the cells):
max_ecDNA_treatment = []
max_HSR_treatment = []

# Max copies for ecDNA and HSRs in post-treatment (after regrowing the population to N_target again):
max_ecDNA_post_treatment = []
max_HSR_post_treatment = []

# Lists for surviving cells 
ePos_survivors = [] # Population counts immediately after treatment (before regrowth) 
eNeg_survivors = []
HSR_survivors = [] 


# Number of cells post-regrowth
ePos_regrown = []
eNeg_regrown = []
HSR_regrown = []

# Iterate loop 50 times and collect data
for i in range(N_runs):
    # Call original (shared ecDNA_dynamics_model) Gillespie loop (pre-treatment)
    population, next_slot, ePos, eNeg, HSR, extinct, *_ = evolve_population(s, N_target, checkpoint_sizes, p_integrate, r, k, p_excision)

    # Runs which produce extinct populations are skipped
    if extinct:
        continue
        
    # -------------------------- pre-treatment: initial population growth --------------------------
    # Record what the population looks like before treatment
    pre_treatment = population[:next_slot].copy()
    
    # Store the highest copy number of ecDNAs + HSRs held in a cell
    max_ecDNA_pre_treatment.append(pre_treatment[:, 0].max())
    max_HSR_pre_treatment.append(pre_treatment[:, 1].max())
    
    # Number of cells in the population before treatment
    ePos_pre_treatment.append(ePos)  
    eNeg_pre_treatment.append(eNeg)
    HSR_pre_treatment.append(HSR)
    
    # -------------------------- treatment: instantaneous weighted kill --------------------------
    # Extract ecDNA and HSR copies from population array 
    ecDNA_copies = population[:next_slot, 0] # ecDNA
    HSR_copies = population[:next_slot, 1] # HSR 

    # Weighted kill where each cell’s probability of death is weighted by its total copy number (1+e+h)
    weighted_copies = 1 + ecDNA_copies + HSR_copies
    # Turn the weighted copies into probabilities (easier than just handling the copies alone) 
    # Do this by dividing a copy in the population over the total sum of the copies 
    weighted_probabilities = weighted_copies / np.sum(weighted_copies)

    # Create an array where all the cells in the population (next_slot) are numbered 
    cell_indices = np.arange(next_slot) 

    # How many deaths in the population - create a variable that defines the size (90%) of the population that will be killed 
    n_kills = int(0.9 * next_slot)

    # Choose which cells to die - even though we are eliminating 90% of the population, the cells to die must have higher copies
    # The index for the ones that are selected to die need to also be noted
    cells_to_kill = rng.choice(cell_indices, size=n_kills, replace=False, p=weighted_probabilities)

    # Need to remove the cells from the population but also keep the surviving 10% 
    surviving_population = np.delete(population[:next_slot], cells_to_kill, axis=0)

    # Max copy numbers --> even though a cell containing HSR and ecDNA is counted as ecDNA+/ePos, this is contributed to the respective max copy number lists
    max_ecDNA_treatment.append(surviving_population[:, 0].max()) 
    max_HSR_treatment.append(surviving_population[:, 1].max())
    
    # Number of cell type survivors - only here in the sub-classification, cell carriers of ecDNA and HSR copies are affected - it will go to ecDNA+ count
    ePos_survivors.append(np.sum((surviving_population[:, 0]) > 0)) # number of ecDNA surviving 
    eNeg_survivors.append(np.sum((surviving_population[:, 0] == 0) & (surviving_population[:, 1] == 0)))
    HSR_survivors.append(np.sum((surviving_population[:, 0] == 0) & (surviving_population[:, 1] > 0))) 
    
    # -------------------------- post-treatment: regrow population --------------------------
    # Call adapted Gillespie loop (evolve_population_after_treatment)
    regrow_population, regrow_next_slot, r_ePos, r_eNeg, r_HSR, r_extinct  = evolve_population_after_treatment(surviving_population, s, N_target, checkpoint_sizes, p_integrate, r, k, p_excision)
    
    # Record after treatment 
    post_treatment = regrow_population[:regrow_next_slot].copy()
    max_ecDNA_post_treatment.append(post_treatment[:, 0].max())
    max_HSR_post_treatment.append(post_treatment[:, 1].max())
    ePos_regrown.append(r_ePos)
    eNeg_regrown.append(r_eNeg)
    HSR_regrown.append(r_HSR)
  

##  ----------------------------------------------- Result 2.1: ecDNA copy-number tail is truncated by treatment and partially rebuilt by growth -----------------------------------------------

# Caclulate standard deviation of ecDNA copy numbers across the highest copies over the 50 replicate runs
std_ecDNA_pre = np.std(max_ecDNA_pre_treatment)
std_ecDNA_treat = np.std(max_ecDNA_treatment)
std_ecDNA_post = np.std(max_ecDNA_post_treatment)

# Sanity check - calculate the average across those standard deviations
print("Pre-treatment max ecDNA (std):", np.mean(max_ecDNA_pre_treatment), "±", std_ecDNA_pre)
print("After treatment max ecDNA (std):", np.mean(max_ecDNA_treatment), "±", std_ecDNA_treat)
print("Post-treatment max ecDNA (std):", np.mean(max_ecDNA_post_treatment), "±", std_ecDNA_post)

# The same for HSR
std_HSR_pre = np.std(max_HSR_pre_treatment)
std_HSR_treat = np.std(max_HSR_treatment)
std_HSR_post = np.std(max_HSR_post_treatment)

print("Pre-treatment max HSR (std):", np.mean(max_HSR_pre_treatment), "±", std_HSR_pre)
print("After treatment max HSR (std):", np.mean(max_HSR_treatment), "±", std_HSR_treat)
print("Post-treatment max HSR (std):", np.mean(max_HSR_post_treatment), "±", std_HSR_post)


# Plotting results 2.1. 
# Define the x-axis. Here we track the highest copy numbers for btoh ecDNA and HSR at Pre-treatment, Post-treatment, and After regrowth of the surviving population
stages = ["Pre-treatment", "Post-treatment", "After regrowth"]

# Caclulate mean and standard deviation of ecDNA copy numbers across the highest copies over the 50 replicate runs
ecDNA_means = [np.mean(max_ecDNA_pre_treatment), np.mean(max_ecDNA_treatment), np.mean(max_ecDNA_post_treatment)]
ecDNA_stds  = [std_ecDNA_pre, std_ecDNA_treat, std_ecDNA_post]

# Caclulate mean and standard deviation of HSR copy numbers across the highest copies over the 50 replicate runs
HSR_means = [np.mean(max_HSR_pre_treatment), np.mean(max_HSR_treatment), np.mean(max_HSR_post_treatment)]
HSR_stds  = [std_HSR_pre, std_HSR_treat, std_HSR_post]

x = np.arange(len(stages))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(x - width/2, ecDNA_means, width, yerr=ecDNA_stds, capsize=5, color="orange", edgecolor="black", label="ecDNA")
ax.bar(x + width/2, HSR_means, width, yerr=HSR_stds, capsize=5, color="red", edgecolor="black", label="HSR")

ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.set_ylabel("Maximum copy number")
ax.set_title("Maximum ecDNA and HSR copy number across treatment stages")
ax.legend()
plt.tight_layout()
plt.show()
