import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Scores
Failure = 2.0
Success = 0.0
Excess = 0.5
Check_data = 1.0

# Load the Excel file
df = pd.read_excel('BR_designstorm_data.xlsx')

# Extract the last two columns
selected_data = df[['bypass_vol_liters', 'precip/design', 'volreduc/design']].dropna()

# Convert the columns to numpy arrays
bypass_array = np.array(selected_data['bypass_vol_liters'])
precip_design_array = np.array(selected_data['precip/design'])
volreduc_design_array = np.array(selected_data['volreduc/design'])

def make_testcases(prop_precip_greater_than_one=0.5, prop_volreduc_greater_than_one=0.5, prop_bypass_one=0.5):
    if not (0 <= prop_precip_greater_than_one <= 1) or not (0 <= prop_volreduc_greater_than_one <= 1) or not (0 <= prop_bypass_one <= 1):
        raise ValueError("Proportion arguments should be between 0 and 1 inclusive")

    size = 100
    
    # For precip_design_array
    cutoff_precip = int(size * prop_precip_greater_than_one)
    precip_high = 1 + np.random.rand(cutoff_precip)
    precip_low = np.random.rand(size - cutoff_precip)
    precip_design_array = np.concatenate([precip_high, precip_low])
    np.random.shuffle(precip_design_array)

    # For volreduc_design_array
    cutoff_volreduc = int(size * prop_volreduc_greater_than_one)
    volreduc_high = 1 + np.random.rand(cutoff_volreduc)
    volreduc_low = np.random.rand(size - cutoff_volreduc)
    volreduc_design_array = np.concatenate([volreduc_high, volreduc_low])
    np.random.shuffle(volreduc_design_array)

    # For bypass_array
    cutoff_bypass = int(size * prop_bypass_one)
    bypass_ones = np.ones(cutoff_bypass)
    bypass_zeros = np.zeros(size - cutoff_bypass)
    bypass_array = np.concatenate([bypass_ones, bypass_zeros]).astype(int)
    np.random.shuffle(bypass_array)
    
    return precip_design_array, volreduc_design_array, bypass_array

precip_design_array, volreduc_design_array, bypass_array = make_testcases(0.15, 0.8, 0.4)

score_array = np.empty(len(precip_design_array))

def calc_score(precip, volreduc, bypass_array):
    upper_flow = 1.2
    lower_flow = 0.8
    scores = np.empty(len(precip))
    for index in range(len(precip)):
        # Bottom right Failure condition
        if precip[index] >= 1.0 and volreduc[index] <= lower_flow:
            scores[index] = Failure
        # Top right Excess condition
        if precip[index] >= 1.0 and volreduc[index] > upper_flow:
            scores[index] = Excess
        # Bottom left Success condition
        if precip[index] <= 1.0 and volreduc[index] <= upper_flow and bypass_array[index] == 0:
            scores[index] = Success
        # Strip Success condition
        if lower_flow <= volreduc[index] <= upper_flow:
            scores[index] = Success
        # Top left Check Data condition
        if precip[index] <= 1.0 and volreduc[index] >= upper_flow:
            scores[index] = Check_data
        # Negative volume check data
        if volreduc[index] < 0.0:
            scores[index] = Check_data
        # Overflow Failure condition
        if precip[index] <= 1.0 and volreduc[index] <= upper_flow and bypass_array[index] != 0:
            scores[index] = Failure
    return scores

score_array = calc_score(precip_design_array, volreduc_design_array, bypass_array)

for ii in range(len(precip_design_array)):
    if score_array[ii] == 0:
        print(precip_design_array[ii], volreduc_design_array[ii], bypass_array[ii], score_array[ii])

average_score = np.mean(score_array)

def new_average(precip_design_array, volreduc_design_array, score_array):
    new_precip = np.copy(precip_design_array)
    new_volreduc = np.copy(volreduc_design_array)

    for ii in range(len(precip_design_array)):
        if score_array[ii] == Success:
            new_precip[ii] = precip_design_array[ii]
            new_volreduc[ii] = volreduc_design_array[ii]
        if score_array[ii] == Failure:
            new_precip[ii] = precip_design_array[ii]*2
        if score_array[ii] == Excess:
            new_volreduc[ii] = volreduc_design_array[ii]*2

    average_precip = np.mean(new_precip)
    average_volreduc = np.mean(new_volreduc)
    return average_precip, average_volreduc

average_precip, average_volreduc = new_average(precip_design_array, volreduc_design_array, score_array)

# average_precip = np.mean(precip_design_array)
# average_volreduc = np.mean(volreduc_design_array)
# print(average_score, average_precip, average_volreduc)

def calc_proportions(score_array):
    # Calculate the proportions of each score
    proportion_failure = np.count_nonzero(score_array == Failure) / len(score_array)
    proportion_success = np.count_nonzero(score_array == Success) / len(score_array)
    proportion_excess = np.count_nonzero(score_array == Excess) / len(score_array)
    proportion_check = np.count_nonzero(score_array == Check_data)/len(score_array)
    return proportion_failure, proportion_success, proportion_excess, proportion_check


# # Create bins based on unique scores, this ensures every score gets a bin
# bins = np.unique(score_array)
# binned_scores = np.digitize(score_array, bins, right=True)

# # Get 4 unique colors from the 'viridis' colormap
# colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, 4)]

# Define a color dictionary
color_dict = {
    Failure: 'red',
    Success: 'green',
    Excess: 'blue',
    Check_data: 'purple'
}

# Map each score to its color
color_map = [color_dict[score] for score in score_array]

# Scatter plot
sc = plt.scatter(precip_design_array, volreduc_design_array, c=color_map, alpha=0.6, edgecolors="w", linewidth=0.5)

# Plot the average point with a star marker
plt.plot(0, 0, '*', color='r', markersize=0, label='Mashup Score: ' + "{:.2f}".format(average_score))
# plt.plot(0, 0, '*', color='r', markersize=0, label='Proportion Check Data: ' + "{:.2f}".format(calc_proportions(score_array)[3]))



# Adding a custom colorbar
from matplotlib.colors import ListedColormap
# Update the custom colorbar to match the new colors
colors = [color_dict[Failure], color_dict[Check_data], color_dict[Excess], color_dict[Success]]
cmap = ListedColormap(colors)
cb = plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap), ticks=[0.125, 0.375, 0.625, 0.875], label='Category Data Proportion')
cb.set_ticklabels([
    'Failure - {:.0f}%'.format(100 * calc_proportions(score_array)[0]),
    'Check Data - {:.0f}%'.format(100 * calc_proportions(score_array)[3]),
    'Excess - {:.0f}%'.format(100 * calc_proportions(score_array)[2]),
    'Success - {:.0f}%'.format(100 * calc_proportions(score_array)[1])
])
# ... [Rest of your plotting code] ...
# Adding horizontal lines
# plt.axhline(y=1.2, color='g', linestyle='--', linewidth=0.8, label='Flow Uncertainty Band')
plt.axhline(y=1.2, color='g', linestyle='--', linewidth=0.8)
plt.axhline(y=0.8, color='g', linestyle='--', linewidth=0.8)

# Titles and labels
plt.title("Hydrologic Performance Index")
plt.xlabel("Precipitation/Est. Design Storm Depth")
plt.ylabel("Volume Retained/Est. Design Volume")
leg = plt.legend(loc='upper left', numpoints = 1, frameon=False, fontsize=12)
plt.grid(True, which="both", ls="--", c='0.7')

# Set x and y axis limits
plt.xlim(-0.5, 5)
plt.ylim(-0.5, 5)

for text in leg.get_texts():
    text.set_weight('bold')

# Show the plot
plt.tight_layout()
plt.show()