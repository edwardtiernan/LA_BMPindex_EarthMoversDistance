import numpy as np
import pyemd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import random 
import math 

## Global variables
counts = 100 # Arbitrary number of BMPs in this set
categories = 5 # Number of distinct operational conditions

# Initialize numpy arrays with extreme BMP classification sets
worst_array = np.full(counts, 1)  # All failing BMPs
best_array = np.full(counts, 5)  # All succeeding BMPs
allthree_array = np.full(counts, 3)  # All Contributing BMPs
twofour_array = np.full(counts, 2)  # Half Subpar and Half Surpassing
for ii in range(len(twofour_array)):
    if ii % 2 == 0:
        twofour_array[ii] = 4
flat_array = np.empty(counts) # Equal representation of all 4 categories
for ii in range(len(flat_array)):
    if ii < counts/categories:
        flat_array[ii] = 1
    if counts/categories <= ii < 2*counts/categories:
        flat_array[ii] = 2
    if 2*counts/categories <= ii < 3*counts/categories:
        flat_array[ii] = 3
    if 3*counts/categories <= ii < 4*counts/categories:
        flat_array[ii] = 4
    if ii >= 4*counts/categories:
        flat_array[ii] = 5

    

# Category Distance Matrix - Distance between the categories is equal to their integer difference
# distance_matrix = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 1.0, 2.0, 3.0], [2.0, 1.0, 0.0, 1.0, 2.0], [3.0, 2.0, 1.0, 0.0, 1.0], [4.0, 3.0, 2.0, 1.0, 0.0]])

# Categorical Lines to Cross - Distance between the categories is equal to the number of operational threshold lines needed to cross
# Threshold lines - Effluent = WQ Threshold, Influent = WQ Threshold, Effluent = Influent
distance_matrix = np.array([[0.0, 2.0, 1.0, 3.0, 3.5], [2.0, 0.0, 3.0, 1.0, 2.5], [1.0, 2.5, 0.0, 2.5, 2.0], [3.0, 1.0, 2.5, 0.0, 0.5], [3.5, 2.5, 2.0, 0.5, 0.0]])

# Compute the Earth Movers Distance using a variable histogram, a target histogram, and a "work / distance" matrix
def compute_EMD(observed_hist, pref_hist, distance_matrix):
    EMD = pyemd.emd(observed_hist, pref_hist, distance_matrix)
    return EMD

# Normalize the Earth Movers Distance score to being from (0, N) where 0 is the worst and N is the number of categories - successful BMPs are given rank N
def normalize_EMDtoAverage(max_EMD, EMD, categories):
    norm_EMD = categories - categories * EMD / max_EMD
    return norm_EMD

# Compute the quintile score - weighted average with frequency where weight is the categorical score 
def compute_quintscore(observed_hist):
    total_bmps = sum(observed_hist)
    proportions = observed_hist[:] / total_bmps
    quint_score = 0
    for prop in range(len(proportions)):
        quint_score = quint_score + proportions[prop] * (prop + 1)
    return quint_score

# Modified average - Categories 2 and 3 are combined into the "Potentially Requiring Managerial Action category"
# After that its a straight up average where Failing = 1, Needs Action = 2, Acceptable = 3, Succeeding = 4
def compute_modifiedaverage(observed_hist):
    total_bmps = sum(observed_hist)
    running_sum = 0
    for cat in range(len(observed_hist)):
        if cat == 0:
            running_sum = running_sum + 1 * observed_hist[cat]
        if cat == 1 or cat == 2:
            running_sum = running_sum + 2 * observed_hist[cat]
        if cat == 3:
            running_sum = running_sum + 4 * observed_hist[cat]
        if cat == 4:
            running_sum = running_sum + 5 * observed_hist[cat]
    mod_average = running_sum/total_bmps
    return mod_average

# This subroutine converts an array of BMP categories into a histogram that can be analyzed
def hist_from_array(input_array, categories):
    hist = np.empty(categories)
    for ii in range(categories):
        num_ii = (input_array == float(ii+1)).sum()
        hist[ii] = num_ii
    return(hist)

# Case study histograms converted from arrays
worst_hist = hist_from_array(worst_array, categories)
best_hist = hist_from_array(best_array, categories)
allthree_hist = hist_from_array(allthree_array, categories)
twofour_hist = hist_from_array(twofour_array, categories)
flat_hist = hist_from_array(flat_array, categories)

# Case study histograms as a list
histograms = [worst_hist, best_hist, allthree_hist, twofour_hist, flat_hist]

# Initialize score lists
raw_EMD_scores = np.empty(len(histograms))
norm_EMD_score = np.empty(len(histograms))
quint_scores = np.empty(len(histograms))
modave_scores = np.empty(len(histograms))

# Call score calculators on each test case histogram
for hh in range(len(histograms)):
    raw_EMD_scores[hh] = compute_EMD(histograms[hh], best_hist, distance_matrix)
    quint_scores[hh] = compute_quintscore(histograms[hh])
    modave_scores[hh] = compute_modifiedaverage(histograms[hh])
    best_EMD = raw_EMD_scores[0]
    norm_EMD_score[hh] = normalize_EMDtoAverage(raw_EMD_scores[0], raw_EMD_scores[hh], categories)
    print(histograms[hh], norm_EMD_score[hh], quint_scores[hh], modave_scores[hh])


rand_raw_EMD_scores = np.empty(counts)
rand_norm_EMD_score = np.empty(counts)
rand_quint_scores = np.empty(counts)
rand_modave_scores = np.empty(counts)

for jj in range(counts):
    # n random floats 
    rand_n = [ random.random() for ii in range(categories) ]
    # extend the floats so the sum is approximately x (might be up to 3 less, because of flooring) 
    bmp_hist = [ math.floor(ii * counts / sum(rand_n)) for ii in rand_n ] 
    # randomly add missing numbers 
    for ii in range(counts - sum(bmp_hist)): 
        bmp_hist[random.randint(0,categories-1)] += 1.0
    # print("The sum is %d" % sum(bmp_hist))
    # print(bmp_hist)
    
    bmp_ints = [float(value) for value in bmp_hist]
    bmp_counts = np.array(bmp_ints)

    # print(bmp_hist, compute_EMD(bmp_counts, pref_counts, distance_matrix), compute_quintscore(bmp_counts))
    rand_raw_EMD_scores[jj] = compute_EMD(bmp_counts, best_hist, distance_matrix)
    rand_quint_scores[jj] = compute_quintscore(bmp_counts)
    rand_modave_scores[jj] = compute_modifiedaverage(bmp_counts)
    rand_norm_EMD_score[jj] = normalize_EMDtoAverage(best_EMD, rand_raw_EMD_scores[jj], categories)
    print(bmp_counts, rand_norm_EMD_score[jj], rand_quint_scores[jj], rand_modave_scores[jj])

# histograms.append(pref_counts)
# EMD_scores.append(compute_EMD(pref_counts, pref_counts, distance_matrix))
# quint_scores.append(compute_quintscore(pref_counts))

plt.scatter(rand_quint_scores, rand_norm_EMD_score, color='k', marker='.', label='Random Sets')
plt.scatter(quint_scores[0], norm_EMD_score[0], color='k', marker='*', s=10, label='All Fails')
plt.scatter(quint_scores[1], norm_EMD_score[1], color='b', marker='*', label='All Succeeds')
plt.scatter(quint_scores[2], norm_EMD_score[2], color='g', marker='*', label='All Contributing')
plt.scatter(quint_scores[3], norm_EMD_score[3], color='c', marker='*', label='Subpar-Surpass Split')
plt.scatter(quint_scores[4], norm_EMD_score[4], color='r', marker='*', label='Equal Distribution')
plt.legend()
plt.xlabel("Quintile Score: 1 (Worst) - 5 (Best)")
plt.ylabel("normalized Earth Mover's Distance: 0 (Worst) - 5 (Best)")
plt.show()

quit()

def from_worst_to_best(curr_array, pref_counts, distance_matrix, categories, max_EMD):

    # curr_hist = hist_from_array(curr_array, categories)
    randomlist = []
    counter = 0
    else_counter = 0
    while counter < 25:
        rr = random.randint(0, counts - 1)
        if ( rr not in randomlist ) and ( curr_array[rr] < categories ):
            randomlist.append(rr)
            counter = counter + 1
        else:
            else_counter = else_counter + 1
            if else_counter > counts:
                print("reached the end")

                new_hist = hist_from_array(curr_array, categories)

                unEMD_score = compute_EMD(new_hist, pref_counts, distance_matrix)
                EMD_score = normalize_EMDtoAverage(max_EMD, unEMD_score, categories)
                quint_score = compute_quintscore(new_hist)
                modave_score = compute_modifiedaverage(new_hist)
                return EMD_score, quint_score, modave_score, new_hist
    # for ii in range(10):
    #     while ( rr not in randomlist ) and ( curr_array[rr] < categories ):
    #         rr = random.randint(0, counts - 1)
    #         randomlist.append(rr)

    for elem in randomlist:
        rr = random.randint(curr_array[elem] + 1, categories)
        curr_array[elem] = rr
    
    # curr_array[randomlist] = curr_array[randomlist] + 1
    new_hist = hist_from_array(curr_array, categories)

    # quit()

    unEMD_score = compute_EMD(new_hist, pref_counts, distance_matrix)
    EMD_score = normalize_EMDtoAverage(max_EMD, unEMD_score, categories)
    quint_score = compute_quintscore(new_hist)
    modave_score = compute_modifiedaverage(new_hist)

    print(EMD_score, quint_score, modave_score)


    return EMD_score, quint_score, modave_score, new_hist


def normalize_EMDtoModAve(EMD_list, modave_list):
    big = max(EMD_list)
    new_EMD_list = [max(modave_list) - max(modave_list) * ii / big for ii in EMD_list]
    return new_EMD_list
    

worst_array = np.full(counts, 1)
worst_hist = hist_from_array(worst_array, categories)

best_array = np.full(counts, 5)
best_hist = hist_from_array(best_array, categories)

histograms = [worst_hist]
EMD_scores = [compute_EMD(worst_hist, best_hist, distance_matrix)]
quint_scores = [compute_quintscore(worst_hist)]
modave_scores = [compute_modifiedaverage(worst_hist)]

max_EMD = EMD_scores[0]

print(normalize_EMDtoAverage(max_EMD, EMD_scores[0], categories), compute_quintscore(worst_hist), compute_modifiedaverage(worst_hist))

curr_array = worst_array

counter = 1

while EMD_scores[-1] < 1.0:
    new_EMD, new_quint, new_modave, new_hist = from_worst_to_best(curr_array, pref_counts, distance_matrix, categories, max_EMD)
    histograms.append(new_hist)
    EMD_scores.append(new_EMD)
    quint_scores.append(new_quint)
    modave_scores.append(new_modave)

    counter = counter + 1
    if counter % 1 == 0:
        print(new_hist)
        plt.scatter(modave_scores, EMD_scores)
        plt.show()


# other = list(reversed(modave_scores))
# other2 = list(reversed(EMD_scores))

print(EMD_scores)
plt.scatter(modave_scores, EMD_scores)
plt.show()