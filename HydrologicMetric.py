import pandas as pd
import numpy as np

# Load the Excel file
df = pd.read_excel('BR_designstorm_data.xlsx')

# Extract the last two columns
selected_data = df.iloc[:, -2:]

# Convert the columns to numpy arrays
precip_design_array = np.array(selected_data['precip/design'])
volreduc_design_array = np.array(selected_data['volreduc/design'])

# Print to verify
print(precip_design_array)
print(volreduc_design_array)