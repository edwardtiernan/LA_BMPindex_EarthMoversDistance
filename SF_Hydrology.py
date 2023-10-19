# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
This script plots the monitoring data from the San Fransisco Green Infrastructure Monitoring Program.
"""

# Set Category Scores
Success = 0
Excess = 1
Check_data = 4
Failure = 10

# Reading the Excel file
df = pd.read_excel('SF_BR_designstorm_data.xlsx', sheet_name='in')

year = None
site = None
# site = "MVGG Rain Gardens"
# site = "Sunset Rain Gardens"
# site = "VVGN Rain Gardens"
# site = "Divisadero BR"
# site = "Broderick St NW BR"


# Remove commas from the "bypass_vol_gal", "precip/design", and "volreduc/design" columns
df['bypass_vol_gal'] = df['bypass_vol_gal'].str.replace(',', '').astype('float64')
# df['precip/design'] = df['precip/design'].str.replace(',', '').astype('float64')
# df['volreduc/design'] = df['volreduc/design'].str.replace(',', '').astype('float64')

# Selecting specific columns from the Excel file
selected_columns = df[['Monitoring Year', 'sitename', 'bypass_vol_gal', 'precip/design', 'volreduc/design']]

# Converting the selected columns to a database
database = selected_columns.to_dict('SF_database')

# Print the datatype of each column in the database
for key, value in database.items():
    print(f"{key}: {type(value[0])}")


def get_columns_by_year_and_site(database, year, site):
    # If year and site are None, return the entire database
    if year is None and site is None:
        return database['bypass_vol_gal'], database['precip/design'], database['volreduc/design']

    # Filter the database to only include rows with the specified year or site
    if year is not None and site is not None:
        filtered_database = {key: [value[i] for i in range(len(value)) if database['Monitoring Year'][i] == year and database['sitename'][i] == site] for key, value in database.items()}
    elif year is not None:
        filtered_database = {key: [value[i] for i in range(len(value)) if database['Monitoring Year'][i] == year] for key, value in database.items()}
    elif site is not None:
        filtered_database = {key: [value[i] for i in range(len(value)) if database['sitename'][i] == site] for key, value in database.items()}

    # Convert the filtered database to numpy arrays
    numpy_arrays = {key: np.array(value) for key, value in filtered_database.items()}

    # Return the numpy arrays for the selected columns
    return numpy_arrays['bypass_vol_gal'], numpy_arrays['precip/design'], numpy_arrays['volreduc/design']


bypass_array, precip_design_array, volreduc_design_array = get_columns_by_year_and_site(database, year, site)


# Calculating the score for each row in the database
def calc_score(precip, volreduc, bypass_array):
    upper_flow = 1.2
    lower_flow = 0.8
    scores = np.empty(len(precip))

    # Set the score for each category
    for ii in range(len(precip)):
        # If the volume reduction is less than zero, Check Data
        if volreduc[ii] < 0.0:
            scores[ii] = Check_data

        # If the volume reduction is greater than the upper volume tolerance, look at the precipitation
        if volreduc[ii] > upper_flow:

            # If the precipitation ratio is greater than or equal to 1.0, Excess
            if precip[ii] >= 1.0:
                scores[ii] = Excess
            
            # If the precipitation ratio is less than 1.0, Check data
            else:
                scores[ii] = Check_data

        # If the volume reduction is less than the lower volume tolerance, look at the precipitation
        elif volreduc[ii] < lower_flow:

            # If the precipitation ratio is greater than or equal to 1.0, Failure
            if precip[ii] >= 1.0:
                scores[ii] = Failure

            # If the precipitation ratio is less than 1.0, look at the bypass
            else:

                # If the bypass array is 0.0, Success
                if bypass_array[ii] == 0.0:
                    scores[ii] = Success

                # Any bypass for precipitation less than 1.0 is a Failure
                else:
                    scores[ii] = Failure

        # Else the volume reduction is between the upper and lower volume tolerances, Success
        else:
            scores[ii] = Success
    
    return scores


scores = calc_score(precip_design_array, volreduc_design_array, bypass_array)
for ss in range(len(scores)):
    print(precip_design_array[ss], volreduc_design_array[ss], bypass_array[ss], scores[ss])


def calc_proportions(score_array):
    # Calculate the proportions of each score
    proportion_failure = np.count_nonzero(score_array == Failure) / len(score_array)
    proportion_success = np.count_nonzero(score_array == Success) / len(score_array)
    proportion_excess = np.count_nonzero(score_array == Excess) / len(score_array)
    proportion_check = np.count_nonzero(score_array == Check_data)/len(score_array)
    return proportion_success, proportion_excess, proportion_check, proportion_failure

   
def plot_data(precip, volreduce, scores, year='', site='All SF GI'):

    # Define a color dictionary
    color_dict = {
        Failure: 'red',
        Success: 'green',
        Excess: 'blue',
        Check_data: 'purple'
    }

    # Map each score to its color
    color_map = [color_dict[score] for score in scores]

    # Scatter plot
    sc = plt.scatter(precip, volreduce, c=color_map, alpha=0.6, edgecolors="w", linewidth=0.5)

    # Plot the average point with a star marker
    average_score = np.mean(scores)
    plt.plot(0, 0, '*', color='r', markersize=0, label='Mashup Score: ' + "{:.2f}".format(average_score))
    # plt.plot(0, 0, '*', color='r', markersize=0, label='Proportion Check Data: ' + "{:.2f}".format(calc_proportions(score_array)[3]))



    # Adding a custom colorbar
    from matplotlib.colors import ListedColormap
    # Update the custom colorbar to match the new colors
    colors = [color_dict[Failure], color_dict[Check_data], color_dict[Excess], color_dict[Success]]
    cmap = ListedColormap(colors)
    cb = plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap), ticks=[0.125, 0.375, 0.625, 0.875], label='Category Data Proportion')
    cb.set_ticklabels([
        'Failure - {:.0f}%'.format(100 * calc_proportions(scores)[3]),
        'Check Data - {:.0f}%'.format(100 * calc_proportions(scores)[2]),
        'Excess - {:.0f}%'.format(100 * calc_proportions(scores)[1]),
        'Success - {:.0f}%'.format(100 * calc_proportions(scores)[0])
    ])
    # ... [Rest of your plotting code] ...
    # Adding horizontal lines
    plt.axhline(y=1.2, color='g', linestyle='--', linewidth=0.8, label='Volume Uncertainty Tolerance')
    # plt.axhline(y=1.2, color='g', linestyle='--', linewidth=0.8)
    plt.axhline(y=0.8, color='g', linestyle='--', linewidth=0.8)

    # Titles and labels
    pltyear = year
    pltsite = site
    if year == None:
        pltyear = 'All Years'
    if site == None:
        pltsite = 'All SF GI'
    plt.title(f"Hydrologic Performance Index (Year {pltyear}, {pltsite})")
    plt.xlabel("Precipitation/Est. Design Storm Depth")
    plt.ylabel("Volume Retained/Est. Design Volume")
    leg = plt.legend(loc='upper left', numpoints = 1, frameon=False, fontsize=12)
    plt.grid(True, which="both", ls="--", c='0.7')

    # Set x and y axis limits
    # plt.xlim(-0.5, 5)
    # plt.ylim(-0.5, 5)

    for text in leg.get_texts():
        text.set_weight('bold')

    # Show the plot
    plt.tight_layout()
    plt.show()
    return

plot_data(precip_design_array, volreduc_design_array, scores, year, site)