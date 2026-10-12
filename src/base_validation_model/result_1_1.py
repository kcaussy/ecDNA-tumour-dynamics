# This .py file presents the code used to generate figure 1.1 for the results 'Base model confirms characterisation of ecDNA copy-number distribution'. 

import numpy as np 
import pandas as pd
rng = np.random.default_rng()
import matplotlib.pyplot as plt

# ----------------------------------------------- Basic model: Selection plot (s=3) ----------------------------------------------- 

# ------------- Copy number segregation function -------------

def CN_seg_Func(N):
    d1 = rng.binomial(2*N, 0.5)
    d2 = 2*N - d1 
    return d1, d2 

# ------------- ecDNA+ population function -------------

def ePos_population_Func(population, next_slot, ePos, eNeg): 
    ePos_idx = np.where((population > 0) & (population != -1))[0]
    idx = rng.choice(ePos_idx)
    dividing_cell = population[idx]
    d1, d2 = CN_seg_Func(dividing_cell)
    if d1 == 0 or d2 == 0: 
        eNeg += 1
    else:
        ePos += 1 
    population[idx] = d1 
    population[next_slot] = d2
    next_slot += 1 
    return population, next_slot, ePos, eNeg


# ------------- Growing the population with segregation events and calculating ecDNA- fraction -------------

def selection_Func(s, N_target, checkpoint_sizes): 
    ePos=1
    eNeg=0
    population = np.full(N_target, -1)  
    population[0] = 1 
    next_slot = 1 
    next_checkpoint_idx = 1 
    eNeg_fraction = [0] 
    
    while next_slot < N_target:
        rnd_ePos = rng.random()
        t1 = -1/(s*ePos) * np.log(rnd_ePos)
        if eNeg > 0:
            rnd_eNeg = rng.random()
            t2 = -1/(eNeg) * np.log(rnd_eNeg)
            if t1 > t2:
                eNeg +=1
                population[next_slot] = 0 
                next_slot += 1 
            else:
                population, next_slot, ePos, eNeg = ePos_population_Func(population, next_slot, ePos, eNeg)
        else:
            population, next_slot, ePos,eNeg = ePos_population_Func(population, next_slot, ePos, eNeg)
        while next_checkpoint_idx < len(checkpoint_sizes) and next_slot >= checkpoint_sizes[next_checkpoint_idx]:
            eNeg_fraction.append(eNeg/next_slot) 
            next_checkpoint_idx += 1
    
    return eNeg_fraction 


# ----------------------------------------------- Result 1.2: Reintegration introduces HSR as a third cell state -----------------------------------------------

# ------------- Storing simulation data -------------
s1_simulation = []
s3_simulation = []
checkpoint_sizes = np.unique(np.logspace(0,5, num=150).astype(int))

# Neutral selection (s=1)
for i in range(100):
    s1_eNeg = selection_Func(s=1, N_target = 10**5, checkpoint_sizes = checkpoint_sizes) 
    s1_simulation.append(s1_eNeg)

# Positive selection (s=3) 
for i in range(100):  
    s3_eNeg = selection_Func(s=3, N_target = 10**5, checkpoint_sizes = checkpoint_sizes) 
    s3_simulation.append(s3_eNeg)

# Average 
s1_avg_sim = np.mean(s1_simulation,axis=0)
s3_avg_sim = np.mean(s3_simulation,axis=0)


# ------------- Reference data -------------
# Load reference data 
s1 = pd.read_csv("ePos_fraction_s=1_lange_et_al.csv")
s3 = pd.read_csv("ePos_fraction_s=3_lange_et_al.csv")

# Extract population size - x-axis
s1_population = s1["population_size"] # Neutral (s=1)
s3_population = s3["population_size"] # Positive (s=3) 

# ------------- Calculate the fraction of ecDNA- cells as the file only handles ecDNA+ -------------

# Extract the data we need to use to work out ecDNA- fraction 
s1_mean_ePos_fraction = s1["mean_ePos_fraction"] 
s3_mean_ePos_fraction = s3["mean_ePos_fraction"] 

s1_eNeg_fraction = 1 - s1_mean_ePos_fraction # 1 - mean ecDNA+ fraction = mean ecDNA- fraction --> author's modelling choice for presenting ecDNA- fraction
s3_eNeg_fraction = 1 - s3_mean_ePos_fraction

# ------------- Standard deviation -------------

s1_sd = s1["std_pos_fraction"]
s3_sd = s3["std_pos_fraction"]

# ------------- Final plot -------------

# Neutral (s1):
plt.semilogx(s1_population, s1_eNeg_fraction, color='grey', label="Reference data: s=1") # Reference data
plt.semilogx(checkpoint_sizes, s1_avg_sim, color="grey", label="Simulation: s=1", linestyle = "dashed") # Simulation
plt.fill_between(s1_population,
                 np.clip(s1_eNeg_fraction - s1_sd), 
                 np.clip(s1_eNeg_fraction + s1_sd, 0, 1), 
                         color="grey", alpha = 0.2) # Standard deviation fill


# Positive (s=3)
plt.semilogx(s3_population, s3_eNeg_fraction, color="blue", label="Reference data: s=3") # Reference data 
plt.semilogx(checkpoint_sizes, s3_avg_sim, color="blue", label="Simulation: s=3", linestyle = "dashed") # Simulation 
plt.fill_between(s3_population,
                 np.clip(s3_eNeg_fraction - s3_sd), 
                 np.clip(s3_eNeg_fraction + s3_sd, 0, 1), 
                         color="blue", alpha = 0.2) # Standard deviation fill 

plt.xticks([1, 10, 100, 1000, 10**4, 10**5],
           ["1", "10", "100", "1000", "$10^4$", "$10^5$"])
plt.xlabel("Number of tumour cells")
plt.ylabel("Frequency of cells without ecDNA")
plt.text(10**4, 0, "shaded = ±1 SD", color = "grey", fontsize = 9)
plt.legend(loc="upper left", prop={"size":7})
plt.title("Mean ecDNA- cell fraction in a tumour population")
plt.show()
