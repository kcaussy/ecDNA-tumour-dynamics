# This .py file presents the code used to generate figure 2.2 for the results 'Treatment reshapes population composition'. 

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


##  ----------------------------------------------- Result 2.2: Treatment reshapes population composition -----------------------------------------------

# Calculate standard deviations for each cell state a each treatment phase 
std_ePos_pre = np.std(ePos_pre_treatment)
std_eNeg_pre = np.std(eNeg_pre_treatment)  # pre-treatment
std_HSR_pre = np.std(HSR_pre_treatment)

std_ePos_post = np.std(ePos_survivors)
std_eNeg_post = np.std(eNeg_survivors) # after treatment
std_HSR_post = np.std(HSR_survivors)

std_ePos_regrowth = np.std(ePos_regrown)
std_eNeg_regrowth = np.std(eNeg_regrown) # after regrowth
std_HSR_regrowth = np.std(HSR_regrown)

# Define the x-axis. Here we track the highest copy numbers for both ecDNA and HSR at Pre-treatment, Post-treatment, and After regrowth of the surviving population
stages = ["Pre-treatment", "Post-treatment", "After regrowth"]

# Calculate the mean population counts across the 50 runs at each phase for each cell state
ePos_means = [np.mean(ePos_pre_treatment), np.mean(ePos_survivors), np.mean(ePos_regrown)]
eNeg_means = [np.mean(eNeg_pre_treatment), np.mean(eNeg_survivors), np.mean(eNeg_regrown)]
HSR_means  = [np.mean(HSR_pre_treatment), np.mean(HSR_survivors), np.mean(HSR_regrown)]

# Combine the standard deviations across the 50 runs at each phase for each cell state
ePos_stds = [std_ePos_pre, std_ePos_post, std_ePos_regrowth]
eNeg_stds = [std_eNeg_pre, std_eNeg_post, std_eNeg_regrowth]
HSR_stds  = [std_HSR_pre, std_HSR_post, std_HSR_regrowth]

# Plot
x = np.arange(len(stages))
width = 0.25   

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, ePos_means, width, yerr=ePos_stds, capsize=4, color="orange", edgecolor="black", label="ecDNA+") # ecDNA+ bar
ax.bar(x, eNeg_means, width, yerr=eNeg_stds, capsize=4, color="steelblue", edgecolor="black", label="ecDNA−") # ecDNA- bar
ax.bar(x + width, HSR_means, width, yerr=HSR_stds, capsize=4, color="red", edgecolor="black", label="HSR") # HSR bar 

ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.set_ylabel("Number of cells")
ax.set_title("Subpopulation composition across treatment stages")
ax.legend()
plt.tight_layout()
plt.show()
