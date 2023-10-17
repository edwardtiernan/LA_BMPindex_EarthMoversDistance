import matplotlib.pyplot as plt
import numpy as np

mashup = {
    "Success": 0.0,
    "Excess": 0.5,
    "Check_data": 1.0,
    "Failure": 2.0
}

# Using a list of lists for test cases for simplicity
test_cases = {
    "SS": [1.0, 0.0, 0.0, 0.0],
    "EE": [0.0, 1.0, 0.0, 0.0],
    "CD": [0.0, 0.0, 1.0, 0.0],
    "FF": [0.0, 0.0, 0.0, 1.0],
    "SE": [0.5, 0.5, 0.0, 0.0],
    "SCD": [0.5, 0.0, 0.5, 0.0],
    "SF": [0.5, 0.0, 0.0, 0.5],
    "ECD": [0.0, 0.5, 0.5, 0.0],
    "EF": [0.0, 0.5, 0.0, 0.5],
    "CDF": [0.0, 0.0, 0.5, 0.5],
    "Eq": [0.25, 0.25, 0.25, 0.25],
    "Eq_NCD": [0.33, 0.33, 0.0, 0.33]
}

# Computing weighted averages for each test case
results = {}

for test_name, weights in test_cases.items():
    weighted_average = sum([weight * value for weight, value in zip(weights, mashup.values())])
    results[test_name] = round(weighted_average, 2)  # Rounding off to 2 decimal places for neatness

# Print out the results
for test_name, average in results.items():
    print(f"{test_name} = {average}")

# # Plotting on the number line
# fig, ax = plt.subplots(figsize=(10, 5))

# # Sort the results for more predictable staggering
# sorted_results = sorted(results.items(), key=lambda x: x[1])

# offset = 0.02
# direction = 1  # initial direction

# for test_name, value in sorted_results:
#     ax.plot(value, 0, 'o', markersize=10)
#     ax.text(value, direction * offset, test_name, ha='center', va='bottom' if direction == 1 else 'top')
    
#     # Flip direction for staggering
#     direction *= -1

# ax.axhline(0, color='black')  # Number line
# ax.set_xlim(0, 2)
# ax.set_ylim(-0.1, 0.3)
# ax.get_yaxis().set_visible(False)  # Hide y axis
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['top'].set_visible(False)

# plt.title("Test Cases on Number Line")
# plt.tight_layout()
# plt.show()


# Plotting on a speedometer-style dial
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

# Convert values to angles in the range [0, 180]
angles = np.array([res * 180 / 2 for res in results.values()])

# Draw the gauge
ax.set_frame_on(False)
ax.set_rticks([])
ax.set_yticklabels([])
ax.set_xticks(np.linspace(0, np.pi, 5))
ax.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0'])
ax.set_ylim(0, 1)

# Plot each value
for angle, (test_name, value) in zip(angles, results.items()):
    ax.plot([0, angle], [0, 0.5], lw=2)
    x_pos = 0.55 * np.cos(angle)
    y_pos = 0.55 * np.sin(angle)
    ax.text(x_pos, y_pos, test_name, ha='center', va='center')

plt.title("Test Cases on Speedometer-style Dial")
plt.show()