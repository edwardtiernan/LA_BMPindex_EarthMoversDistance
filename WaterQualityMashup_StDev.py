import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import itertools
from adjustText import adjust_text

# Water Quality Weights for old threshold score
success_1 = 0.0
excess_1 = 0.5
insufficient_1 = 1.0
marginal_1 = 2.5
failure_1 = 3.5

# Water quality weights for new threshold score
success_2 = 0.0
excess_2 = 0.5
insufficient_2 = 2.0
marginal_2 = 1.5
failure_2 = 5.0

# Water quality weights for Rank10 threshold score
success_3 = 0.0
excess_3 = 1
insufficient_3 = 4
marginal_3 = 3
failure_3 = 10

# Number of BMPs
bmps = 50
categories = 5

def create_array(a, b, c, d, e):
    # Given numbers
    numbers = [1, 2, 3, 4, 5]
    
    # Number of times each number needs to be repeated
    repeats = [a, b, c, d, e]
    
    # Create an array by repeating the numbers the specified number of times
    array = np.array([num for num, rep in zip(numbers, repeats) for _ in range(rep)])
    
    return array

# Arrays for different distributions
FF_array = create_array(50, 0, 0, 0, 0)  # All failure
MM_array = create_array(0, 50, 0, 0, 0)  # All marginal 
II_array = create_array(0, 0, 50, 0, 0)  # All insufficient
EE_array = create_array(0, 0, 0, 50, 0)  # All excess
SS_array = create_array(0, 0, 0, 0, 50)  # All success
Eq20_array = create_array(12, 12, 12, 12, 12) # Equal representation of all 5 categories
FS50_array = create_array(25, 0, 0, 0, 25) # Half Failure and Half Success
FE50_array = create_array(25, 0, 0, 25, 0)  # Half Failure and Half Excess
FI50_array = create_array(25, 0, 25, 0, 0)  # Half Failure and Half Insufficient
FM50_array = create_array(25, 25, 0, 0, 0)  # Half Failure and Half Marginal
MS50_array = create_array(0, 25, 0, 0, 25)  # Half Marginal and Half Success
ME50_array = create_array(0, 25, 0, 25, 0)  # Half Marginal and Half Excess
MI50_array = create_array(0, 25, 25, 0, 0)   # Half Marginal and Half Insufficient
IS50_array = create_array(0, 0, 25, 0, 25)   # Half Insufficient and Half Success
IE50_array = create_array(0, 0, 25, 25, 0)   # Half Insufficient and Half Excess
ES50_array = create_array(0, 0, 0, 25, 25)   # Half Excess and Half Success

# This function calculates the score for each test case
def calc_testcase_score(array, success, excess, insufficient, marginal, failure):
    score = np.zeros(len(array))
    for ii in range(len(array)):
        if array[ii] == 1:
            score[ii] = failure
        elif array[ii] == 2:
            score[ii] = marginal
        elif array[ii] == 3:
            score[ii] = insufficient
        elif array[ii] == 4:
            score[ii] = excess
        elif array[ii] == 5:
            score[ii] = success
    return score


def generate_coordinates(num_points_FF, num_points_MM, num_points_II, num_points_EE, num_points_SS):
    """
    Generates a numpy array of random 2D coordinates within different ranges 
    and conditions and then appends them together.

    Parameters:
    num_points_per_set (int): Number of random coordinates to generate per set.

    Returns:
    numpy.ndarray: A 2D numpy array where each row represents a coordinate (x, y).
    """
    
    # Generate FAILURE set
    def generate_set_1(num_points):
        coordinates = np.empty((num_points, 2))
        for i in range(num_points):
            y = 1.0 + np.random.rand()  # Ensure 1.0 < y < 2.0
            x = np.random.uniform(0, min(y + 0.01, 2.0))  # Ensure x < y and 0.0 < x < 2.0
            coordinates[i] = [x, y]
        return coordinates
    
    # Generate MARGINAL set
    def generate_set_2(num_points):
        coordinates = np.empty((num_points, 2))
        for i in range(num_points):
            y = np.random.rand()  # Ensure 0.0 < y < 1.0
            x = np.random.uniform(0, y - 0.01)  # Ensure x < y
            coordinates[i] = [x, y]
        return coordinates
    
    # Generate INSUFFICIENT set
    def generate_set_3(num_points):
        coordinates = np.empty((num_points, 2))
        for i in range(num_points):
            y = 1 + np.random.rand()  # Ensure 1.0 < y < 2.0
            x = np.random.uniform(max(y, 1.0), 2.0)  # Ensure x > y and 1.0 < x < 2.0
            coordinates[i] = [x, y]
        return coordinates
    
    # Generate EXCESS set
    def generate_set_4(num_points):
        coordinates = np.empty((num_points, 2))
        for i in range(num_points):
            x = np.random.rand()  # Ensure 0.0 < x < 1.0
            y = np.random.uniform(0, x)  # Ensure y < x
            coordinates[i] = [x, y]
        return coordinates
    
    # Generate SUCCESS set
    def generate_set_5(num_points):
        coordinates = np.empty((num_points, 2))
        for i in range(num_points):
            x = 1 + np.random.rand()  # Ensure 1.0 < x < 2.0
            y = np.random.uniform(0, min(x, 1.0))  # Ensure y < x and 0.0 < y < 1.0
            coordinates[i] = [x, y]
        return coordinates

    # Generating different sets of coordinates
    set_1 = generate_set_1(num_points_FF)
    set_2 = generate_set_2(num_points_MM)
    set_3 = generate_set_3(num_points_II)
    set_4 = generate_set_4(num_points_EE)
    set_5 = generate_set_5(num_points_SS)

    # Appending them together
    all_coordinates = np.vstack((set_1, set_2, set_3, set_4, set_5))

    return all_coordinates


# This function creates a random distribution of points within the range [0, 2)
def generate_random_coordinates(num_points, proportion_x_below_one=0.5, proportion_y_below_one=0.5):
    """
    Generates a numpy array of random 2D coordinates within the range [0, 2).
    The proportion of x and y coordinates below 1.0 are adjustable.

    Parameters:
    num_points (int): Number of random coordinates to generate.
    proportion_x_below_one (float): Proportion of x coordinates below 1.0. Should be between 0 and 1.
    proportion_y_below_one (float): Proportion of y coordinates below 1.0. Should be between 0 and 1.

    Returns:
    numpy.ndarray: A 2D numpy array where each row represents a coordinate (x, y).
    """
    if not (0 <= proportion_x_below_one <= 1) or not (0 <= proportion_y_below_one <= 1):
        raise ValueError("Proportions should be between 0 and 1")
    
    x_coordinates = np.concatenate([
        np.random.rand(int(num_points * proportion_x_below_one)),
        1 + np.random.rand(num_points - int(num_points * proportion_x_below_one))
    ])
    np.random.shuffle(x_coordinates)

    y_coordinates = np.concatenate([
        np.random.rand(int(num_points * proportion_y_below_one)),
        1 + np.random.rand(num_points - int(num_points * proportion_y_below_one))
    ])
    np.random.shuffle(y_coordinates)

    coordinates = np.vstack((x_coordinates, y_coordinates)).T
    return coordinates

# This function evaluates the coordinates and assigns a score to each
def evaluate_coords(coords, success, excess, insufficient, marginal, failure):
    score = np.zeros(len(coords))
    for ii in range(len(coords)):
        x = coords[ii][0]
        y = coords[ii][1]
        if x >= 1 and y <= 1:
            score[ii] = success
        elif x <= 1 and y <= 1 and x >= y:
            score[ii] = excess
        elif x >= 1 and y > 1 and x >= y:
            score[ii] = insufficient
        elif x < 1 and y < 1 and x < y:
            score[ii] = marginal
        else:
            score[ii] = failure
        print(f"({x}, {y}) = {score[ii]}")
    return score

# Calculate the Index score (weighted average) and standard deviation of a test case
def calc_mode_stdev(array):

    # New array for StDev
    mode_array = np.empty(len(array))

    # Determine the mode
    mode = stats.mode(array, keepdims=True)[0][0]

    for ii in range(len(array)):
        if array[ii] == mode:
            mode_array[ii] = 0
        else:
            mode_array[ii] = 1

    # Calculate the standard deviation
    stdev = np.average(mode_array)
    return stdev


def plot_coordinates(coordinates, scores, color_dict, average, stdev):
    """
    Plots a scatter plot of the given 2D coordinates.

    Parameters:
    coordinates (numpy.ndarray): A 2D numpy array where each row represents a coordinate (x, y).
    """
    # Validate the inputs
    if len(coordinates) != len(scores):
        raise ValueError("The lengths of coordinates and scores must be equal.")

    # Extracting x and y coordinates
    x = [coordinate[0] for coordinate in coordinates]
    y = [coordinate[1] for coordinate in coordinates]

    # Creating a scatter plot
    for i, score in enumerate(scores):
        plt.scatter(x[i], y[i], color=color_dict.get(score, 'k'))  # 'k' is the default color (black) if the score is not found in the dictionary

    # Adding an invisible data point for the legend
    # plt.scatter([], [], color='none', edgecolor='none', label=f'10.0 (Bad) - 0.0 (Good) Mash-Up Score: {average} \u00B1 {stdev}')
    plt.scatter([], [], color='none', edgecolor='none', label=f'10.0 (Bad) - 0.0 (Good) Mash-Up Score: {average}')


    # Set the x-axis and y-axis limits to be from 0 to 2
    plt.xlim(0, 2)
    plt.ylim(0, 2)

    # Adding black dotted lines
    plt.plot([1, 1], [0, 1], color='k', linestyle='dotted')  # Adjusted to stop at y=1
    plt.axhline(y=1, color='k', linestyle='dotted')
    plt.plot([0, 2], [0, 2], color='k', linestyle='dotted')  # x=y line

    plt.legend()
    plt.xlabel('Influent/Threshold Concentration')
    plt.ylabel('Effluent/Threshold Concentration')
    plt.title('Water Quality Performance Index')
    plt.show()
    return plt


def main():

    # Plot the contrived test case scores on a scatter plot with _1 vs _2
    test_cases = [FF_array, MM_array, II_array, EE_array, SS_array, Eq20_array, FS50_array, FE50_array, FI50_array, FM50_array, MS50_array, ME50_array, MI50_array, IS50_array, IE50_array, ES50_array]
    scores_1 = np.zeros(len(test_cases))
    scores_3 = np.zeros(len(test_cases))
    for test_case in range(len(test_cases)):
        score_1 = calc_testcase_score(test_cases[test_case], success_1, excess_1, insufficient_1, marginal_1, failure_1)
        score_3 = calc_testcase_score(test_cases[test_case], success_2, excess_2, insufficient_2, marginal_2, failure_2)

        scores_1[test_case] = round(np.average(score_1), 2)
        scores_3[test_case] = round(np.average(score_3), 2)

    # Define marker_cycler with distinct markers
    marker_cycler = itertools.cycle(["o", "v", "^", "<", ">", "s", "+", "x", ".", "*"])
    label_cycler = ["FF", "MM", "II", "EE", "SS", "Eq20", "FS50", "FE50", "FI50", "FM50", "MS50", "ME50", "MI50", "IS50", "IE50", "ES50"]

    # plt.scatter(rand_quint_scores, rand_norm_EMD_score, color='k', marker='.', label='Random Sets')
    # plt.figure(figsize=(10, 6))

    # texts = []  # To store all the annotations
    # for hh, test_case in enumerate(test_cases):

    #     # Use hh to access the corresponding scores and label, and use next(marker_cycler) to get the next marker in the cycle.
    #     x, y = scores_3[hh], scores_1[hh]
    #     plt.scatter(x, y, marker=next(marker_cycler))
        
    #     # # Annotate each point with the corresponding label from label_cycler
    #     plt.annotate(label_cycler[hh], (x, y), textcoords="offset points", xytext=(0, 10), ha='center')    

    #     # Annotate each point with the corresponding label from label_cycler
    #     texts.append(plt.text(x, y, label_cycler[hh], ha='center', va='center'))
    
    # adjust_text(texts)
    
    # plt.legend()
    # plt.xlabel("Threshold Cross New: 3.5 (Worst) - 0.0 (Best)")
    # plt.ylabel("Threshold Cross Old: 3.5 (Worst) - 0.0 (Best)")
    # plt.grid(True)
    # plt.show()


    # Plot the randomly generated test cases and their scores on the WQ Framework
    for ii in range(5):
        # coords = generate_coordinates(12, 26, 4, 35, 23) # Int'l Database
        coords = generate_coordinates(0, 0, 50, 0, 50) # F, M, I, E, S
        # coords = generate_random_coordinates(50, 0.6, 0.4)
        score_array = evaluate_coords(coords, success_3, excess_3, insufficient_3, marginal_3, failure_3)

        average = round(np.average(score_array), 2)
        stdev = round(calc_mode_stdev(score_array), 2)  


        # Define a color dictionary
        color_dict = {
            failure_1: 'red',
            failure_2: 'red',
            failure_3: 'red',
            marginal_1: 'orange',
            marginal_2: 'orange',
            marginal_3: 'orange',
            insufficient_1: 'yellow',
            insufficient_2: 'yellow',
            insufficient_3: 'yellow',
            excess_1: 'blue',
            excess_2: 'blue',
            excess_3: 'blue',
            success_1: 'green',
            success_2: 'green',
            success_3: 'green'
        }

        # Example Usage:
        # coordinates = np.array([[0.5, 1.5], [1.2, 0.8], [1.5, 1.9], [0.7, 0.3]])
        plot_coordinates(coords, score_array, color_dict, average, stdev)

main()

