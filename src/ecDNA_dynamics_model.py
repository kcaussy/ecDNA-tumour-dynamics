"""
ecDNA_dynamics_model.py

This is a stochastic model for ecDNA/HSR tumour dynamics and is used as the basis for 
all models (untreated and treatment) in this project. It implements a Gillespie birth-death 
# process to grow the population, where each cell is represented as an (ecDNA, HSR) copy-number pair. 

The model contains the following functions:
    CN_seg_Func              - copy-number segregation (random for ecDNA, faithful for HSR)
    ecDNA_divide_Func        - ecDNA+ cell division, including reintegration
    HSR_Func                 - HSR cell division, including excision
    cell_to_die_Func         - background cell death (shared across all three states)
    evolve_population        - the Gillespie loop tying the above together

And is used by: simulations/untreated_model.py and simulations/treatment_model.py
"""


import numpy as np
rng = np.random.default_rng(seed=None)

# ------------- Copy number segregation function -------------

# ecDNA and HSR copy number segregation during cell division.
# A parent cell with N ecDNA copies first undergo replication, producing 2N copies. 
# Each replicated copy is then independently assigned to one of the two daughter cells with 
# probability 0.5, resulting in a binomial distribution of copy numbers. 
# This reflects the acentromeric nature of ecDNA, whereby copies segregate randomly rather than 
# equally between daughter cells, meaning one daughter will receive either the rest or no copies.
# HSR, by contrast, is integrated into the chromosome and therefore segregates faithfully. both daughters 
# inherit the parent's HSR count. 

def CN_seg_Func(e, h): # e = ecDNA, h = HSR cell 
    d1_e = rng.binomial(2*e, 0.5)
    d2_e = 2*e - d1_e 
    d1_h = h
    d2_h = h 
    # Return each daughter cell into pairs: ecDNA + HSR  (easier to unpack)
    return (d1_e, d1_h), (d2_e, d2_h)

# ------------- ecDNA+ cell fate function -------------

# Simulates the fate of a dividing ecDNA+ cell by modelling ecDNA reintegration 
# and daughter cell classification. 
# All cells are stored in a 2D np.full array (in the Gillespie loop).
# An ecDNA+ cell is randomly selected to divide (identified through column 0 of the population array, 
# where ecDNA copy number is stored). This function handles that division: potential reintegration,
# segregation, and classification of the resulting daughters.

def ecDNA_divide_Func(population, next_slot, ePos, eNeg, p_integrate, HSR):
    ePos_idx = np.where(population[:next_slot, 0] > 0)[0]
    idx = rng.choice(ePos_idx)
    dividing_cell = population[idx]
    e, h = dividing_cell 

# Reintegration event: A per-copy event where each e copy independently has probability
# p_integrate of integrating into the chromosome per cell division event (pre-replication). 
# Resultantly, copies must move from the ecDNA pool to the HSR pool. 
    n_integrate = rng.binomial(e, p_integrate)
    e = e - n_integrate 
    h = h + n_integrate 

# The cell divides with the integrated copies, the parent ecDNA+ cell is 
# subtracted from the subpopulation  
    d1, d2 = CN_seg_Func(e, h) 
    ePos -= 1  

# Classify the daughters: 

# Daughter 1: 
    # If there are ecDNA copy numbers greater than 0 then the daughter is classed as ecDNA+
    # d1[0] > 0 =  ecDNA+
    if d1[0] > 0:
        ePos +=1 # grow the ecDNA+ population 

    # If it doesnt, such as the daughter has no ecDNA copies BUT has HSR copies, it is classed as HSR
    # d1[0] == 0 and d1[1] > 0 | HSR
    elif d1[0] == 0 and d1[1] > 0:
        HSR += 1 # grow the HSR population
        
    # If the daughter has neither (there are no ecDNA copy numbers and HSR copies), it is classed as ecDNA-  
    # d1[0] == 0 and d1[1] == 0
    else: 
        eNeg += 1 # grow the ecDNA- population
        
# Repeat for daughter 2:   
    if d2[0] > 0:
        ePos +=1 
    elif d2[0] == 0 and d2[1] > 0:
        HSR += 1
    else: 
        eNeg += 1  

    # Write the daughters into the array
    population[idx] = d1 # parent slot will be overwritten by daughter 1
    population[next_slot] = d2 # daughter 2 goes into the next empty slot
    next_slot += 1 # move the pointer by 1 
    
    return population, next_slot, ePos, eNeg, HSR, n_integrate


# ------------- HSR cell division with ecDNA excision function -------------

# HSR segregation event during cell division. The dividing cell starts as a pure HSR cell. 
# If excision does not occur, division is faithful (HSR only). If excision occurs, one copy reverts 
# to ecDNA and segregates randomly, while remaining HSR copies still segregate faithfully. 

def HSR_Func(population, next_slot, HSR, eNeg, ePos, p_excision): 
    excision_event = False 
    HSR_idx = np.where((population[:next_slot, 1] > 0) & (population[:next_slot, 0] == 0))[0]
    HSR_slot = rng.choice(HSR_idx)
    e, h = population[HSR_slot]

    # Excision event: a single per-cell probability check (not per-copy). If h > 0 and the check succeeds
    # one representative copy is released from the HSR array back to an ecDNA status, regardless of the total HSR count.
    # If excision occurs, one HSR copy converts to ecDNA on the parent before segregation - the actual effect on population
    # counts is then determined by segregation and daughter classification below. 
  
    if h > 0 and rng.random() < p_excision:
        excision_event = True 
        e += 1
        h -= 1

    # Segregation is consistent, CN_seg_Func is called to carry out the segregation 
    # of either ecDNA and/or HSR copies. 
    # ecDNA will segregate randomly (binomial) and HSR will segregate faithfully.
    d1, d2 = CN_seg_Func(e, h)

    # Remove parent from HSR count, replaced by one of its daughters 
    HSR -= 1

# Classify the daughters: 

  # Daughter 1: 
    # If the daughters e > 0, increase the ePos+ count by 1 
    # - the HSR cell now has 'x' ecDNA+ copies: 
    if d1[0] > 0:
        ePos += 1 
      
    # Else, if daughter 1 has 0 ecDNA copies and daughter 2 has HSR copies - this is a pure HSR cell 
    # increase the HSR count by 1
    elif d1[0] == 0 and d1[1] > 0: 
        HSR += 1
      
    # If they aren't either, it is an ecDNA- cell, increase the eNeg counter by 1
    else:
        eNeg += 1

# Daughter 2:   
    # we do the same for daughter 2 
    if d2[0] > 0:
        ePos +=1 
    elif d2[0] == 0 and d2[1] > 0:
        HSR += 1
    else: 
        eNeg += 1  
        
    # write the daughters in     
  
    population[HSR_slot] = d1
    population[next_slot] = d2
    next_slot += 1
    
    return population, next_slot, HSR, eNeg, ePos, excision_event 

# ------------- cell death function -------------

# A cell is randomly chosen to die from a pre-selected subpopulation (ePos, eNeg, or HSR 
# - determined by the calling function). This function represents background cell death and is reused
# across all three cell states. 

def cell_to_die_Func(cell_death_idx, next_slot, population):
    # Cell is randomly chosen to die
    death_idx = rng.choice(cell_death_idx)
  
    # Remove the cell chosen to die at said index from the population by replacing it with a
    # live cell ([next_slot -1] takes the last live cell added to the population)
    population[death_idx] = population[next_slot -1] 
  
    # Reduce the total live population count by 1 (the specific subpopulation counter - ePos, 
    # eNeg, or HSR - is updated separately by the calling function) 
    next_slot -= 1
    return next_slot, population


# ------------- Gillespie loop: population evolution -------------

# The Gillespie loop allows population growth where at each step, competing exponential waiting times are drawn for every possible
# birth and death event across the three cell states (ecDNA+, ecDNA-, and HSR)' the shortest waiting time determines which event
# fires next. The loop continues until the population reaches N_target, tracking reintegration and excision events, and periodically 
# recording subpopulation fractions and ecDNA copy number distributions at specified checkpoints. 

def evolve_population(s, N_target, checkpoint_sizes, p_integrate, r, k, p_excision): 
    ePos=1
    eNeg=0
    HSR = 0 
    population = np.full((N_target, 2), -1) 
    population [0] = [1, 0] 
    next_slot = 1 
    extinct = False 
    next_checkpoint_idx = 0
    event_excision = []
    event_reintegration = []
    eNeg_fraction = [] 
    ePos_fraction = []
    HSR_fraction = []
    M1 = []
    M2 = []

  
# The loop iterates until the population reaches N_target, or exits and reports extinction if all three
# threesubpopulations reach zero (filters out extinct runs)
    while next_slot < N_target and not extinct:

# A division time and death rate is calculated: 
        
        # ------------- calculate division times (tb1, tb2, tb3) -------------
        # Calculate the division times for each population. If there aren't any cells in the population, then the 
        # division time is set to infinity (np.inf()) 
        # e.g. t1 is calculated if the ePos population is greater than 0, if it is not division time = infinity 
        tb1 = -1/(s*ePos) * np.log(rng.random()) if ePos > 0 else np.inf # ecDNA+ division time 
        tb2 = -1/(eNeg) * np.log(rng.random()) if eNeg > 0 else np.inf # ecDNA- division time 
        tb3 = -1/(r*HSR) * np.log(rng.random()) if HSR > 0 else np.inf # HSR division time 

        
        # ------------- calculate death rates (td1, td2, td3) -------------
        # Cell death is modelled as proportional to birth (death rate = k x birth rate), so death scales with the birth
        # rate of each specific state. Because the same k applies to every state, it doesn't distort the fitness differences
        # set by s and r - it scales overall turnover equally across all three cell types. 
        td1 = -1/(k*s*ePos) * np.log(rng.random()) if ePos > 0 else np.inf # ecDNA+ death 'time'
        td2 = -1/(k*eNeg) * np.log(rng.random()) if eNeg > 0 else np.inf # ecDNA- death 'time' 
        td3 = -1/(k*r*HSR) * np.log(rng.random()) if HSR > 0 else np.inf # HSR death 'time'

# Then both the division time and death rate are compared to see which event occurs:
      
        # ------------- calculate shortest rate/ time -------------
        # Calculate the smallest birth (division) time (SBT), the shortest division time distinguishes what cell divides first
        SBT = min(tb1, tb2, tb3)
        # Calculate the smallest death 'time' (SDT), the shortest death time distinguishes what cell dies first
        SDT = min(td1, td2, td3)

        
        # Compare both rates/times, whichever is the smallest is the one to occur:
        # Is the smallest birth time less than the smallest death time? if so the winning subpopulation divides
        if SBT < SDT: 
            
            # ------------- Scenario 1a: ecDNA+ cell divides -------------
            # Does SBT = t1? if so, ecDNA+ (ePos) cell divides, call ecDNA_divide_Func
            if SBT == tb1:
                population, next_slot, ePos, eNeg, HSR, n_integrate = ecDNA_divide_Func(population, next_slot, ePos, eNeg, p_integrate, HSR)
                 # if the shortest birth/division time is an ecDNA and inside the function an reintegration event has happened - record the event
                if n_integrate > 0:
                    event_reintegration.append(np.log2(next_slot))
           
            # ------------- Scenario 2a: ecDNA- cell divides -------------
            # If not, does SBT = t2? then ecDNA- (eNeg) cell divides, no function called, add ecDNA- cell to population 
            # and advance pointer in array
            elif SBT == tb2: 
                eNeg +=1
                population[next_slot] = [0, 0] 
                next_slot += 1    
        
            # ------------- Scenario 3a: HSR cell divides -------------
            # If neither, SBT = t3 and an HSR cell divides. Call HSR_Func() 
            else: 
                population, next_slot, HSR, eNeg, ePos, excision_event = HSR_Func(population, next_slot, HSR, eNeg, ePos, p_excision)
                # If a HSR is chosen to divide and inside the function an excision event occurs - record the event 
                if excision_event:
                    event_excision.append(np.log2(next_slot))
        
        
        # If not the corresponding subpopulation with the smallest death rate will have a cell randomly chosen to die:
        else:
            # ------------- Scenario 1b: ecDNA+ cell dies -------------
            # If the shortest death rate is td1, an ecDNA+ cell is randomly chosen to die and will be removed from the total population. 
            # The cell removed will be replaced with the last live cell in the population array (of any type), and 1 is removed from the ePos
            # count specifically
            if SDT == td1:
                # Identify all the cells (ecDNA+/ePos) in the population so a cell can be randomly chosen to die
                cell_death_idx = np.where((population[:next_slot, 0] > 0))[0]
                next_slot, population = cell_to_die_Func(cell_death_idx, next_slot, population) # call cell_to_die function
                ePos -= 1
              
            # ------------- Scenario 2b: ecDNA- cell dies -------------
            # If the shortest death rate is not td1 but td2, an ecDNA- cell is randomly chosen to die
            elif SDT == td2:
                cell_death_idx = np.where((population[:next_slot, 1] == 0) & (population[:next_slot, 0] == 0))[0]
                # Call function for eNeg cell to die 
                next_slot, population = cell_to_die_Func(cell_death_idx, next_slot, population) # call cell_to_die function
                eNeg -= 1 
              
            # ------------- Scenario 3b: HSR cell dies -------------
            # If the shortest time is neither td1 or td2, a HSR cell will be chosen to die as td3 has the shortest death rate
            else: 
                cell_death_idx = np.where((population[:next_slot, 1] > 0) & (population[:next_slot, 0] == 0))[0]
                next_slot, population = cell_to_die_Func(cell_death_idx, next_slot, population) # call cell_to_die function
                HSR -=1
                
            # An extinct population is classed as having no cells in its population count
            extinct = (ePos== 0 and eNeg == 0 and HSR == 0)

        # ------------- Calculating subpoulation fractions of ecDNA-, mean and variance -------------
        # While there are still checkpoints to record and the population has reached the next checkpoint size (checkpoint sizes defined when 
        # testing function) record a snapshot of the ecDNA copy number distribution
      
        if next_checkpoint_idx < len(checkpoint_sizes) and next_slot >= checkpoint_sizes[next_checkpoint_idx]:
            # extract the ecDNA copy numbers from all occupied slots in the population array 
            N = population[:next_slot, 0] 
            eNeg_fraction.append(eNeg/next_slot)
            ePos_fraction.append(ePos/next_slot)
            HSR_fraction.append(HSR/next_slot)
            M1.append(np.mean(N)) 
            M2.append(np.var(N)) 
            next_checkpoint_idx += 1
    
    return population, next_slot, ePos, eNeg, HSR, extinct, event_excision, event_reintegration, eNeg_fraction, ePos_fraction, HSR_fraction, M1, M2   






