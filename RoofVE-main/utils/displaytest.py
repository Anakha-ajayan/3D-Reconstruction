import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

# Load the CSV file
csv_file = r"C:\Users\akhil\OneDrive\Documents\point2building\RoofVE-main\pred_res\100_1struc.csv"
csv_data = pd.read_csv(csv_file)

# Print column names to verify
print("Columns in CSV:", csv_data.columns)

# Extract relevant columns from CSV
csv_points_x = csv_data['id_x']
csv_points_y = csv_data['id_y']
csv_points_z = csv_data['id_z']


# Load TXT data
txt_file = r"C:\Users\akhil\OneDrive\Documents\point2building\RoofVE-main\pred_res\100_finCor_Geo.txt"
txt_data = pd.read_csv(txt_file, sep=",", header=None, names=["x", "y", "z"])

# Extract 3D points from TXT
txt_points_x = txt_data["x"]
txt_points_y = txt_data["y"]
txt_points_z = txt_data["z"]

# Visualization
fig = plt.figure(figsize=(14, 7))

# Subplot 1: CSV data
ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(csv_points_x, csv_points_y, csv_points_z, c="blue", marker="o", label="CSV Points")
ax1.set_title("3D Points from CSV")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()

# Subplot 2: TXT data
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(txt_points_x, txt_points_y, txt_points_z, c="red", marker="^", label="TXT Points")
ax2.set_title("3D Points from TXT")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.legend()

# Show plots
plt.tight_layout()
plt.show()

