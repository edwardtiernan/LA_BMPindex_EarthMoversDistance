import numpy as np
import pyemd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from cycler import cycler

import random 
import math 

## Global variables
counts = 10000 # Number of sets to assess striation
bmps = 100 # Number of BMPs in the set
categories = 5 # Number of distinct operational conditions

# Initialize numpy arrays with test-case BMP classification sets
intl_array = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, \
             4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, \
             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]) # Copper scores from Int'l database on bioretention
FF_array = np.full(bmps, 1)  # All failure
MM_array = np.full(bmps, 2)  # All marginal 
II_array = np.full(bmps, 3)  # All insufficient
EE_array = np.full(bmps, 4)  # All excess
SS_array = np.full(bmps, 5)  # All success
Eq20_array = np.empty(bmps) # Equal representation of all 5 categories
for ii in range(len(Eq20_array)):
    if ii < bmps/categories:
        Eq20_array[ii] = 1
    if bmps/categories <= ii < 2*bmps/categories:
        Eq20_array[ii] = 2
    if 2*bmps/categories <= ii < 3*bmps/categories:
        Eq20_array[ii] = 3
    if 3*bmps/categories <= ii < 4*bmps/categories:
        Eq20_array[ii] = 4
    if ii >= 4*bmps/categories:
        Eq20_array[ii] = 5
FS50_array = np.full(bmps, 1) # Half Failure and Half Success
for ii in range(len(FS50_array)):
    if ii % 2 == 0:
        FS50_array[ii] = 5
ME50_array = np.full(bmps, 2)  # Half Marginal and Half Excess
for ii in range(len(ME50_array)):
    if ii % 2 == 0:
        ME50_array[ii] = 4
MI50_array = np.full(bmps, 2)   # Half Marginal and Half Insufficient
for ii in range(len(MI50_array)):
    if ii % 2 == 0:
        MI50_array[ii] = 3

SE50_array = np.full(bmps, 5)   # Half Success and Half Excess
for ii in range(len(SE50_array)):
    if ii % 2 == 0:
        SE50_array[ii] = 4

EI50_array = np.full(bmps, 4)   # Half Excess and Half Insufficient
for ii in range(len(EI50_array)):
    if ii % 2 == 0:
        EI50_array[ii] = 3

MF50_array = np.full(bmps, 2)   # Half Marginal and Half Failure
for ii in range(len(MF50_array)):
    if ii % 2 == 0:
        MF50_array[ii] = 1

"""
Distance Matrices

The following largely commented section contains various distance matrices for the threshold cross score (calculated by the Earth Mover's Distance algorithm).
Distance arrays are determined by summing the threshold weights needed to cross from one category to each other.  Various distance matrices are created for 
variable threshold weighting schemes.
"""
# Category Distance Matrix - Distance between the categories is equal to their integer difference
# distance_matrix = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 1.0, 2.0, 3.0], [2.0, 1.0, 0.0, 1.0, 2.0], [3.0, 2.0, 1.0, 0.0, 1.0], [4.0, 3.0, 2.0, 1.0, 0.0]])

# Categorical Lines to Cross - Distance between the categories is equal to the number of operational threshold lines needed to cross
# Weights: Effluent = 2, Export = 1, Influent = 0.5
# distance_matrix = np.array([[0.0, 2.0, 1.0, 3.0, 3.5], [2.0, 0.0, 3.5, 1.0, 1.5], [1.0, 3.5, 0.0, 2.5, 2.0], [3.0, 1.0, 2.5, 0.0, 0.5], [3.5, 1.5, 2.0, 0.5, 0.0]])

# Weights: Effluent = 1, Export = 2, Influent = 0.5
distance_matrix = np.array([[0.0, 1.0, 2.0, 3.0, 3.5], [1.0, 0.0, 3.5, 2.0, 2.5], [2.0, 3.5, 0.0, 1.5, 1.0], [3.0, 2.0, 1.5, 0.0, 0.5], [3.5, 2.5, 1.0, 0.5, 0.0]])

# Weights: Effluent = 1, Export = 2, Influent = 0.5, allowing for asymmetric traveling
# distance_matrix = np.array([[0.0, 1.0, 2.0, 3.0, 3.0], [1.0, 0.0, 3.0, 2.0, 2.5], [2.0, 3.5, 0.0, 1.5, 1.0], [3.0, 2.0, 1.5, 0.0, 0.5], [3.5, 2.5, 1.0, 0.5, 0.0]])

# Categorical Lines to Cross - Distance between the categories is equal to the number of operational threshold lines needed to cross
# Weights: Effluent = 3, Export = 2, Influent = 1
# distance_matrix = np.array([[0.0, 3.0, 2.0, 5.0, 5.0], [3.0, 0.0, 6.0, 2.0, 3.0], [2.0, 6.0, 0.0, 4.0, 3.0], [5.0, 2.0, 4.0, 0.0, 1.0], [5.0, 3.0, 3.0, 1.0, 0.0]])

# Ken's Weights: Effluent = 2, Export = 1, Influent = 1
# distance_matrix = np.array([[0.0, 2.0, 1.0, 3.0, 3.0], [2.0, 0.0, 4.0, 1.0, 2.0], [1.0, 4.0, 0.0, 3.0, 2.0], [3.0, 1.0, 3.0, 0.0, 1.0], [3.0, 2.0, 2.0, 1.0, 0.0]])


"""
Compute Functions
"""

# Compute the Earth Movers Distance using a variable histogram, a target histogram (All Success), and a distance matrix
def compute_EMD(observed_hist, pref_hist, distance_matrix):
    EMD = pyemd.emd(observed_hist, pref_hist, distance_matrix)
    return EMD

# Normalize the Earth Movers Distance score to being from (0, N) where 0 is the All Success score and N is equal to the sum of threshold weights needed to cross from Failure to Success
def normalize_EMDtoAverage(max_EMD, EMD, categories):
    norm_EMD =  np.max(distance_matrix) * EMD / max_EMD
    # norm_EMD = 5 - 4*EMD/max_EMD
    return norm_EMD

# Compute the quintile score - weighted average with frequency where weight is the categorical score 
def compute_quintscore(observed_hist):
    total_bmps = sum(observed_hist)
    proportions = observed_hist[:] / total_bmps
    quint_score = 0
    for prop in range(len(proportions)):
        quint_score = quint_score + proportions[prop] * (prop + 1)
    # return quint_score

    # Reverse the quint score so it trends with the TCPI
    quint_score_r = categories + 1 - quint_score
    return quint_score_r

# This subroutine converts an array of BMP categories into a histogram that can be analyzed
def hist_from_array(input_array, categories):
    hist = np.empty(categories)
    for ii in range(categories):
        num_ii = (input_array == float(ii+1)).sum()
        hist[ii] = num_ii
    return(hist)


"""
Main
"""

def main():
    # Case study histograms converted from arrays
    intl_hist = hist_from_array(intl_array, categories)
    FF_hist = hist_from_array(FF_array, categories)
    MM_hist = hist_from_array(MM_array, categories)
    II_hist = hist_from_array(II_array, categories)
    EE_hist = hist_from_array(EE_array, categories)
    SS_hist = hist_from_array(SS_array, categories)
    Eq20_hist = hist_from_array(Eq20_array, categories)
    FS50_hist = hist_from_array(FS50_array, categories)
    ME50_hist = hist_from_array(ME50_array, categories)
    MI50_hist = hist_from_array(MI50_array, categories)
    SE50_hist = hist_from_array(SE50_array, categories)
    EI50_hist = hist_from_array(EI50_array, categories)
    MF50_hist = hist_from_array(MF50_array, categories)

    # Case study histograms as a list
    histograms = [FF_hist, MM_hist, II_hist, EE_hist, SS_hist, Eq20_hist, FS50_hist, ME50_hist, MI50_hist, SE50_hist, EI50_hist, MF50_hist, intl_hist]

    # Initialize score lists
    raw_EMD_scores = np.empty(len(histograms))
    norm_EMD_score = np.empty(len(histograms))
    quint_scores = np.empty(len(histograms))
    modave_scores = np.empty(len(histograms))

    # Call score calculators on each test case histogram
    for hh in range(len(histograms)):
        raw_EMD_scores[hh] = compute_EMD(histograms[hh], SS_hist, distance_matrix)
        quint_scores[hh] = compute_quintscore(histograms[hh])
        worst_EMD = raw_EMD_scores[0]
        norm_EMD_score[hh] = normalize_EMDtoAverage(worst_EMD, raw_EMD_scores[hh], categories)
        print(histograms[hh], f'{norm_EMD_score[hh]:.3}', quint_scores[hh], modave_scores[hh])


    rand_raw_EMD_scores = np.empty(counts)
    rand_norm_EMD_score = np.empty(counts)
    rand_quint_scores = np.empty(counts)

    marker_cycler = ["o", "v", "^", "<", ">", "s", "+", "x", ".", "*", "*", "*", "*"]
    label_cycler = ["FF", "MM", "II", "EE", "SS", "Eq20", "FS50", "ME50", "MI50", "SE50", "EI50", "MF50", "Int'l Database"]

    f = plt.figure()
    ax = f.add_subplot(111)
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")
    for hh in range(len(histograms)):
        plt.scatter(quint_scores[hh], norm_EMD_score[hh], marker=marker_cycler[hh], label=label_cycler[hh])
    plt.legend()
    plt.xlabel("Quint Average: 1.0 (Best) - 5.0 (Worst)")
    plt.ylabel("Threshold Cross: 0.0 (Best) - 3.5 (Worst)")
    plt.show()
    return

main()
