# This .py file presents the code used to generate figure 1.1 for the results 'Base model confirms characterisation of ecDNA copy-number distribution'. 

# ---------------------------- Basic model: growing population & copy number distribution ----------------------------

import numpy as np 
import matplotlib.pyplot as plt
rng = np.random.default_rng()

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

# ------------- Growing the population with segregation events -------------
ePos=1
eNeg=0 
s=1 
N_target=10**4
population = np.full(N_target, -1) 
population[0] = 1 
next_slot = 1 

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
            population, next_slot, ePos, eNeg = ePos_population_Func(population, next_slot,ePos, eNeg)
    else:
        population, next_slot, ePos,eNeg = ePos_population_Func(population, next_slot,ePos,eNeg)

print(population)
print(f"Number of ecDNA+ cells: {ePos}")
print(f"Number of ecDNA- cells: {eNeg}")

# --------------------------- Result 1.1: Base model confirms characterisation of ecDNA copy-number distribution ---------------------------
CN_values = population

# ecDNA+ population is determined by the number of cells that have copy numbers greater than 0 
ePos_CN = CN_values[CN_values > 0]

# ecDNA- population is determined by cells that have 0 copy numbers 
eNeg_CN = CN_values[CN_values == 0]

# Dispaly the bars that correspond to ecDNA+ cells 
bins = np.arange(0, ePos_CN.max() + 2, 1)  

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ------------- Left: ecDNA+ distribution only -------------
ax1.hist(ePos_CN, bins=bins, alpha=0.7, color="orange", edgecolor="black", label="ecDNA+ copy number")
ax1.set_xlabel("ecDNA copy number per cell")
ax1.set_ylabel("Frequency")
ax1.set_title("ecDNA+ Copy Number Distribution")
ax1.legend()

# ------------- Right: population counts as a bar chart -------------
ax2.bar(["ecDNA+", "ecDNA-"], [ePos, eNeg], color=["orange", "steelblue"], alpha=0.7, edgecolor="black")
ax2.set_ylabel("Frequency")
ax2.set_title("ecDNA+ vs ecDNA− Population Counts")

for copies, val in enumerate([ePos, eNeg]):
    ax2.text(copies, val + 50, str(val), ha="center", fontweight="bold")

plt.suptitle("ecDNA Copy Number Distribution", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
 
