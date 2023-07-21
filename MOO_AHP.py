import ahpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# # Test Case - N = 4, sequential ranking
# categories = ['TSS', 'TN', 'Cu', 'FIB']
# TCPI = [3.5, 3.5, 0, 0]
# rankings = [1, 2, 3, 4]
# marker_cycler = ["o", "v", "^", "<"]
# label_cycler = ["TSS = " + str(rankings[0]), "TN = " + str(rankings[1]), \
#                 "Cu = " + str(rankings[2]), "FIB = " + str(rankings[3]),]

# # Test Case - N = 4, extreme ranking
# categories = ['TSS', 'TN', 'Cu', 'FIB']
# TCPI = [3.5, 3.5, 3.5, 0]
# rankings = [1, 4, 4, 4]
# marker_cycler = ["o", "v", "^", "<"]
# label_cycler = ["TSS = " + str(rankings[0]), "TN = " + str(rankings[1]), \
#                 "Cu = " + str(rankings[2]), "FIB = " + str(rankings[3]),]


# # Test Case - N = 6, sequential ranking
# categories = ['TSS', 'TN', 'TP', 'Cu', 'Zn', 'FIB']
# rankings = [1, 2, 3, 4, 5, 6]
# TCPI = [3.5, 3.5, 3.5, 0, 0, 0]
# marker_cycler = ["o", "v", "^", "<", ">", "x"]
# label_cycler = ["TSS = " + str(rankings[0]), "TN = " + str(rankings[1]), "TP = " + str(rankings[2]), \
#                 "Cu = " + str(rankings[3]), "Zn = " + str(rankings[4]), "FIB = " + str(rankings[5]),]

# Test Case - N = 6, sequential ranking
categories = ['TSS', 'TN', 'TP', 'Cu', 'Zn', 'FIB']
TCPI = [0.5, 2.5, 2.5, 1.0, 1.0, 3.5]
rankings = [1, 2, 2, 5, 5, 6]
marker_cycler = ["o", "v", "^", "<", ">", "x"]
label_cycler = ["TSS = " + str(rankings[0]), "TN = " + str(rankings[1]), "TP = " + str(rankings[2]), \
                "Cu = " + str(rankings[3]), "Zn = " + str(rankings[4]), "FIB = " + str(rankings[5]),]


# Initialize an empty dictionary to store the combinations and rankings
comp_ratios = {}

# Loop through the categories and rankings to create the combinations
for i in range(len(categories)):
    for j in range(i + 1, len(categories)):
        category1 = categories[i]
        category2 = categories[j]
        ranking1 = rankings[i]
        ranking2 = rankings[j]
        
        # Calculate the combination ratio and add it to the dictionary
        comp_ratios[(category1, category2)] = float(ranking2)/float(ranking1)

print(comp_ratios)
# quit()

# Comparison ratios are the 2nd categories ranking / 1st categories ranking
# comp_ratios = {('TSS', 'TN'): 2/1, ('TSS', 'TP'): 3/1, ('TSS', 'Cu'): 4/1, ('TSS', 'Zn'): 5/1, ('TSS', 'FIB'): 6/1, \
#                ('TN', 'TP'): 3/2, ('TN', 'Cu'): 4/2, ('TN', 'Zn'): 5/2, ('TN', 'FIB'): 6/2, \
#                ('TP', 'Cu'): 4/3, ('TP', 'Zn'): 5/3, ('TP', 'FIB'): 6/3, \
#                ('Cu', 'Zn'): 5/4, ('Zn', 'FIB'): 6/4, \
#                ('Zn', 'FIB'): 6/5}

def weighted_average(categories, rankings, TCPI):
    ahp_weights_ranksum = []
    numR = len(categories)
    # print(numR)
    sumR = sum(rankings)
    for rank in range(len(rankings)):
        rankprop = (numR + 1 - rankings[rank])/(sumR)
        ahp_weights_ranksum.append(rankprop)
    return ahp_weights_ranksum

ahp_weights_ranksum = weighted_average(categories, rankings, TCPI)

ahp_weights = ahpy.Compare(name="Index", comparisons=comp_ratios, precision = 3, random_index = 'saaty')

ahp_list = list(ahp_weights.target_weights.values())

perf_weights_ahp = []
perf_weights_ranksum = []
for cat in range(len(categories)):
    perf_weights_ahp.append(TCPI[cat] * ahp_list[cat])
    perf_weights_ranksum.append(TCPI[cat] * ahp_weights_ranksum[cat])

performance_index_AHP = sum(perf_weights_ahp)
performance_index_RankSum = sum(perf_weights_ranksum)
print(performance_index_AHP, performance_index_RankSum)

print(ahp_list)
print(ahp_weights_ranksum)
# print(ahp_list)

def radar_chart(categories, values, title, color_map = 'viridis'):
    # Calculate angle for each category
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

    # Ensure the plot is a full circle
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Use colormap to map values to colors
    cmap = plt.get_cmap(color_map)
    colors = cmap(np.array(values) / max(values))

    # Draw each segment with its own color
    for i in range(len(values) - 1):
        color = colors[i]
        # Create a polygon and add it to the plot
        polygon = patches.Polygon([(angles[i], 0), (angles[i], values[i]), 
                                    (angles[i+1], values[i+1]), (angles[i+1], 0)], closed=True, color=color)
        ax.add_patch(polygon)

    # Draw line from the center to each category and add labels
    plt.xticks(angles[:-1], categories)

    # Add markers on top of the colored area
    for angle, value, color in zip(angles, values, colors):
        ax.plot([angle], [value], marker='o', markersize=5, color='black')

    # Fix axis to go from 0 to max value
    ax.set_yticklabels([])
    ax.set_rlabel_position(30)

    plt.title(title)
    plt.show()

radar_chart(categories, TCPI, "TCPI Scores by Pollutant Type")

# print(ahp_weights.consistency_ratio)

# f = plt.figure()
# ax = f.add_subplot(111)
# # ax.yaxis.tick_right()
# # ax.yaxis.set_label_position("right")
# for cat in range(len(categories)):
#     plt.scatter(ahp_weights_ranksum[cat], ahp_list[cat], marker=marker_cycler[cat], label=label_cycler[cat])
# plt.legend()
# plt.title("AHP vs Rank Sum Proportions for Ranked Pollutant Species")
# plt.xlabel("Rank Sum Proportions")
# plt.ylabel("Analytic Hierarchy Process Proportions")
# plt.xticks(np.arange(0, len(ahp_weights_ranksum)/sum(rankings)+0.01, 1/sum(rankings)))
# # plt.text(0.25, 0.85, 'N = 6', transform = ax.transAxes) # Coordinates and text
# # plt.text(0.25, 0.80, 'Sequential Ranking', transform = ax.transAxes)
# # def format_func(value, tick_number):
# #     # find number of multiples of pi/2
# #     N = len(rankings)
# #     for nn in range(N):
# #         return str(ahp_list[nn])
# # ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
# plt.show()
