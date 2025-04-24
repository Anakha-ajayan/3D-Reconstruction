import os
import random
import numpy as np
import open3d as o3d
import utils.evaluation as eval_func
import sys

# Add the path to the utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

# Set paths
PLY_FOLDER = r'C:\Users\akhil\OneDrive\Documents\point2building\RoofVE-main\pointcloud'  # Path to .ply files
GT_FOLDER = r'C:\Users\akhil\OneDrive\Documents\point2building\RoofVE-main\wireframe'  # Update to your .obj files
GT_START = 'v'

# Select a random .ply file
ply_files = [f for f in os.listdir(PLY_FOLDER) if f.endswith('.xyz')]
if not ply_files:
    raise FileNotFoundError(f"No .ply files found in {PLY_FOLDER}")
random_ply = random.choice(ply_files)
random_ply_path = os.path.join(PLY_FOLDER, random_ply)

# Corresponding GT file (assuming the filename matches)
roof_id = os.path.splitext(random_ply)[0]  # Assuming filenames match
gt_file = os.path.join(GT_FOLDER, GT_START + str(roof_id) + '.obj')

if not os.path.exists(gt_file):
    raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

# Load data
pred_pcd = o3d.io.read_point_cloud(random_ply_path)  # Load .ply file
vs_pred = np.asarray(pred_pcd.points)  # Predicted points (m, 3)

# Function to extract vertices from .obj file
def extract_vertices_from_obj(file_path):
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Line starts with 'v' indicates a vertex
                # Extract the vertex coordinates (x, y, z)
                vertex = list(map(float, line.strip().split()[1:]))  # Skip 'v' and get coordinates
                vertices.append(vertex)
    return np.array(vertices)

# Extract ground truth vertices from .obj file
vs_gt = extract_vertices_from_obj(gt_file)

# Step 1: Find corresponding predicted points for each ground truth point
v_gt_pred_dist = eval_func.find_v_gt_pred_dist(vs_gt, vs_pred)

# Step 2: Calculate true positives (TP), total ground truth, and predictions
vs_tp, vs_tp_num, vs_gt_num, vs_pred_num = eval_func.stats_tp_gt_pred(v_gt_pred_dist, vs_gt, vs_pred, thrs_true=1)

# Step 3: Precision and Recall
precision, recall = eval_func.calc_p_r(vs_tp_num, vs_gt_num, vs_pred_num)

# Step 4: Calculate VD (x, y, z)
vd_x, vd_y, vd_z = eval_func.calc_vd_xyz(vs_tp)

# Print results
print("Testing file:", random_ply)
print("Precision:", precision)
print("Recall:", recall)
print("VD (x, y, z):", vd_x, vd_y, vd_z)

# Step 5: Visualize
gt_pcd = o3d.geometry.PointCloud()
gt_pcd.points = o3d.utility.Vector3dVector(vs_gt)

tp_pcd = o3d.geometry.PointCloud()
tp_pcd.points = o3d.utility.Vector3dVector(vs_tp[:, :3])  # GT coordinates of TP

pred_pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (vs_pred.shape[0], 1)))  # Predicted (red)
gt_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (vs_gt.shape[0], 1)))  # Ground truth (green)
tp_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (vs_tp.shape[0], 1)))  # True Positives (blue)

# Visualize in Open3D
o3d.visualization.draw_geometries([pred_pcd, gt_pcd, tp_pcd])

