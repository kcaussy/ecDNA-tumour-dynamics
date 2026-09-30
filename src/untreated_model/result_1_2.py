# Packages
import numpy as np 
import matplotlib.pyplot as plt

# Import functions from shared model
from ecDNA_dynamics_model import evolve_population

# This .py file presents the code used to generate figure 1.2 for the results 'Reintegration introduces HSR as a third cell state'. 


# ------------- Result 1.2: Reintegration introduces HSR as a third cell state -------------

# Set parameters 
s = 1.5
r = 1.2
k = 0.3 # test value
N_target = 10**5
p_integrate = 0.001
checkpoint_sizes = [1, 100, 500, 1000, 2000, 5000, 10000, 10**5]

# Call evolve_population function
population, next_slot, ePos, eNeg, HSR, extinct, event_excision, event_reintegration, eNeg_fraction, ePos_fraction, HSR_fraction, M1, M2 = evolve_population(s, N_target, checkpoint_sizes, p_integrate, r, k, p_excision)

# Extract copy numbers for each cell type 
ecDNA_CN = population[:next_slot, 0]
ePos_CN = ecDNA_CN[ecDNA_CN > 0]

# In the simulation, the possibility of extinct runs can arise. As we are displaying the ecDNA+ copy number distribution, we need 
# to make sure that there are ecDNA+ cells in the population
# If there aren't any ecDNA+ cells then we display the message:
if ePos_CN.size == 0: 
    print("No ecDNA+ cells found at this checkpoint. Set a different checkpoint or parameter.")
else: 
    # If there are, then set bins for the plot:
    bins = np.arange(0, ePos_CN.max() + 2, 1)  
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ------------- Left: ecDNA+ distribution only -------------
    ax1.hist(ePos_CN, bins=bins, alpha=0.7, color="orange", edgecolor="black", label="ecDNA+ copy number")
    ax1.set_xlabel("ecDNA+ copy number per cell")
    ax1.set_ylabel("Frequency")
    ax1.set_title("ecDNA+ Copy Number Distribution")
    ax1.legend()

# ------------- Right: population counts per cell state -------------
    ax2.bar(["ecDNA+", "ecDNA-", "HSR"], [ePos, eNeg, HSR], color=["orange", "steelblue", "red"], alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Frequency")
    ax2.set_title("ecDNA+ vs ecDNA− vs HSR Population Counts")


for copies, val in enumerate([ePos, eNeg, HSR]):
    ax2.text(copies, val + 50, str(val), ha="center", fontweight="bold")

plt.suptitle("ecDNA Copy Number Distribution", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# Output: src/untreated_model/figures/population_composition.png 



