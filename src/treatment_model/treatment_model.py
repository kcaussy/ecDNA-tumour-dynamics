# Packages
import numpy as np 
import matplotlib.pyplot as plt
rng = np.random.default_rng()

# Import functions from shared model
from ecDNA_dynamics_model import evolve_population

# This .py file presents the code used to generate the simulation results for the treatment model. ]
# Full walkthrough of the treatment protocol can be found here in src/treatment_model/treatment_model_protocol.md. 

 # --------------------------------------------------------------------

# Set parameters 
s = 1.5
r = 1.2 
k = 0.1
N_runs = 50
N_target = 10**5
p_integrate = 0.01
p_excision = 0.02
checkpoint_sizes = [100, 500, 1000, 2000, 5000, 10**4]

# --- Phase 1 - initial growth of population ---
# Run evolve_population to give us the current population size (should be whatever N_target is)
population, next_slot, ePos, eNeg, HSR, extinct, *_ = evolve_population(s, N_target, checkpoint_sizes, p_integrate, r, k, p_excision)


# --- Phase 2 - simulating weighted (instantaneous) kill ---
# Extract ecDNA and HSR copies from population array 
ecDNA_copies = population[:next_slot, 0] # ecDNA
HSR_copies = population[:next_slot, 1] # HSR 

# Death probability proportional to total copy number
weighted_copies = 1 + ecDNA_copies + HSR_copies

# Convert weighted copies into probabilities (sums to 1) by dividing each cell's weight
# by the total weight across the population
weighted_probabilities = weighted_copies / np.sum(weighted_copies)

# Index every cell in the population so we have something to sample from
cell_indices = np.arange(next_slot) 

# 90% of the population will be killed
n_kills = int(0.9 * next_slot)

# Randomly select which cells die, weighted so higher-copy cells are more likely to be chosen
# (without replacement, so each cell can only be selected once)
cells_to_kill = rng.choice(cell_indices, size=n_kills, replace=False, p=weighted_probabilities)

# Remove the killed cells from the population, leaving the surviving 10%
surviving_population = np.delete(population[:next_slot], cells_to_kill, axis=0)


# --- Phase 3 - Post-treatment regrowth ---
def evolve_population_after_treatment(surviving_population, s, N_target, checkpoint_sizes, p_integrate, r, k, p_excision): 

# Initialise the simulation using the surviving post-treatment population rather than a single found ecDNA+ cell 
    ePos= np.sum(surviving_population[:, 0] > 0)
    eNeg= np.sum((surviving_population[:, 0] == 0) & (surviving_population[:, 1] == 0))
    HSR = np.sum((surviving_population[:, 0] ==0) & (surviving_population[:, 1] > 0))
    population = np.full((N_target, 2), -1)

# Define how many surviving cells there are (10,000)
    n_survivors = len(surviving_population)

# Select the first row of the new array which holds the surviving population and bring to front of array
    population[:n_survivors] = surviving_population 
    next_slot = n_survivors
    extinct = False 

    while next_slot < N_target and not extinct:
        
# ------------- calculate division times (tb1, tb2, tb3) ------------

        tb1 = -1/(s*ePos) * np.log(rng.random()) if ePos > 0 else np.inf   # ecDNA+ division time 
        tb2 = -1/(eNeg) * np.log(rng.random()) if eNeg > 0 else np.inf     # ecDNA- division time
        tb3 = -1/(r*HSR) * np.log(rng.random()) if HSR > 0 else np.inf     # HSR division time

# ------------- calculate death rates (td1, td2, td3) -------------

        td1 = -1/(k*s*ePos) * np.log(rng.random()) if ePos > 0 else np.inf   # ecDNA+ death 'time'
        td2 = -1/(k*eNeg) * np.log(rng.random()) if eNeg > 0 else np.inf     # ecDNA- death 'time'
        td3 = -1/(k*r*HSR) * np.log(rng.random()) if HSR > 0 else np.inf     # HSR death 'time'

# ------------- calculate shortest rate/ time -------------
        SBT = min(tb1, tb2, tb3)
        SDT = min(td1, td2, td3)

        if SBT < SDT:  
# ------------- scenario 1a: ecDNA+ cell divides -------------
            if SBT == tb1:
                population, next_slot, ePos, eNeg, HSR, n_integrate = ecDNA_divide_Func(population, next_slot, ePos, eNeg, p_integrate, HSR)

# ------------- scenario 2a: ecDNA- cell divides -------------
            elif SBT == tb2: 
                eNeg +=1
                population[next_slot] = [0, 0] 
                next_slot += 1         

# ------------- scenario 3a: HSR cell divides -------------
            else: 
                population, next_slot, HSR, eNeg, ePos, excision_event = HSR_Func(population, next_slot, HSR, eNeg, ePos, p_excision)
                    
        else:     
# ------------- scenario 1b: ecDNA+ cell dies -------------
            if SDT == td1:
                cell_death_idx = np.where((population[:next_slot, 0] > 0))[0]
                next_slot, population = cell_to_die_Func(cell_death_idx, next_slot, population)
                ePos -= 1

# ------------- scenario 2b: ecDNA- cell dies -------------
            elif SDT == td2:
                cell_death_idx = np.where((population[:next_slot, 1] == 0) & (population[:next_slot, 0] == 0))[0]
                next_slot, population = cell_to_die_Func(cell_death_idx, next_slot, population)
                eNeg -= 1 

# ------------- scenario 3b: HSR cell dies -------------
            else: 
                cell_death_idx = np.where((population[:next_slot, 1] > 0) & (population[:next_slot, 0] == 0))[0]
                next_slot, population = cell_to_die_Func(cell_death_idx, next_slot, population)
                HSR -=1
                
            extinct = (ePos== 0 and eNeg == 0 and HSR == 0)
        
    return population, next_slot, ePos, eNeg, HSR, extinct


# --------------------------------------------------------------------

# Simulating treatment protocol over 50 replicate runs (data collection): 

# Initialise lists:

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
