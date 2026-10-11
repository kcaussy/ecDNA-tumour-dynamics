# ---------------------------- Basic model: growing population & copy number distribution ----------------------------

import numpy as np 
rng = np.random.default_rng()

# ------------- Copy number segregation function -------------
# Simulate random segregation of ecDNA copies during division. 

def CN_seg_Func(N):
    d1 = rng.binomial(2*N, 0.5)
    d2 = 2*N - d1 
    return d1, d2 

# ------------- ecDNA+ population function -------------
# Handles one division of a randomly chosen ecDNA+ (ePos) cell --> a dividing ePos cell is replaced by its two daughters 
# the population grows by net +1

def ePos_population_Func(population, next_slot, ePos, eNeg): 
    # Filter through the array -> to identify the ecDNA+ cells it must have a copy numer greater than 0 and not equal to -1 
    ePos_idx = np.where((population > 0) & (population != -1))[0]
    # Pick a random ecDNA+ cell to divide, with every run a random ecDNA+ cell will always divide, not the same one
    idx = rng.choice(ePos_idx)
    dividing_cell = population[idx] # Record at what index the chosen cell has been made to divide and distribute the ecDNA 
    # Copy numbers between the daughter cells of the parent cell cell that was chosen 
    d1, d2 = CN_seg_Func(dividing_cell)
    # If either daughter inherts 0 copies, that daughter is a new ecDNA- cell 
    # Simulate both cases as CN_seg_Func can return 0 copies to either daughter 1 or 2 
    if d1 == 0 or d2 == 0: 
        eNeg += 1
    else:
        ePos += 1 
    # Write the daughters into the array
    population[idx] = d1 # Parent slot will be overwritten by daughter 1
    population[next_slot] = d2 # Daughter 2 goes into the next empty slot 
    next_slot += 1 # Move the pointer by 1 
    return population, next_slot, ePos, eNeg


# ------------- Growing the population with segregation events -------------
# Gillespie stochastic simiultion

# Initial state --> tumour starts as one ecDNA+ cell with 1 ecDNA copy, no ecDNA- cells are present
ePos=1
eNeg=0 
s=1 
N_target=10**4
population = np.full(N_target, -1) # np.full replaces list & np.zeros. it takes the desired population 
                                   # Size and empty slot placeholders as -1 
population[0] = 1 # Founding cell --> 1 ecDNA+ copy 
next_slot = 1 # Next empty slot (also current population size)

# next_slot = current population size (as it is responsible for adding a new cell into the array), as long as this is less than
# the target population (N_target) the loop will continue  
while next_slot < N_target:
    rnd_ePos = rng.random()
    t1 = -1/(s*ePos) * np.log(rnd_ePos) # t1 = ecDNA+ division time
    # If the population of ecDNA- cells is more than 0 (in each run) we calculate t2
    if eNeg > 0:
        rnd_eNeg = rng.random()
        t2 = -1/(eNeg) * np.log(rnd_eNeg) # ecDNA- division time
        # Comparison for shorter division times: if ecDNA+ division time is more than ecDNA- then an ecDNA- cell divides

        if t1 > t2:
            # Only an ecDNA+ cell can produce ecDNA- cells through random segregation of its copy numbers. Therefore, when we model
            # an ecDNA- cell dividing the two daughters will be ecDNA-: 
            # daughter 1 (0 copies) replaces parent cell as it already holds 0 copies
            # daughter 2 (0 copies) is added to the next slot so there is a net gain of +1 
            eNeg +=1
            population[next_slot] = 0 
            next_slot += 1 # Move pointer by 1 (as one daughter cell has been added) 
        # Inner loop, t1 < t2: ecDNA+ cell divides first, call function to simulate segregation event. this determines whether an ecDNA+ or ecDNA- daughter is produced 
        else:
            population, next_slot, ePos, eNeg = ePos_population_Func(population, next_slot,ePos, eNeg)
            
    # Outside loop: call function again. If there are no ecDNA- cells in the population an ecDNA+ cell divides automatically and the segregation event occurs
    else:
        population, next_slot, ePos,eNeg = ePos_population_Func(population, next_slot,ePos,eNeg)
