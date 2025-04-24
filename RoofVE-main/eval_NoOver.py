import os
import numpy as np
import pandas as pd
import utils.evaluation as eval_func  # Ensure this module is correctly implemented

# Paths to your folders (update accordingly)
GT_FOLDER = r'C:\Users\akhil\OneDrive\Documents\point2building\RoofVE-main\wireframe'  # Folder containing .obj files
PRED_FOLDER = r'C:\Users\akhil\OneDrive\Documents\point2building\RoofVE-main\pointcloud'  # Folder containing .xyz files
SAVE_FOLDER = './res/structure_line_exp/NoOver_Md/'

GT_END = '.obj'
PRED_END = '.xyz'
isSave_2_csv = True

# Save results
save_name = os.path.join(SAVE_FOLDER, 'eval_res_' + os.path.basename(os.path.dirname(PRED_FOLDER)) + '_mGT.csv')

# List all .xyz files and extract filenames without extensions
xyz_files = [f for f in os.listdir(PRED_FOLDER) if f.endswith(PRED_END)]
roof_ids = [os.path.splitext(f)[0] for f in xyz_files]  # Extract base filenames

columns_name = ['roof_id', 'precision', 'recall', 'vd_x', 'vd_y', 'vd_z',
                'vs_tp_num', 'vs_gt_num', 'vs_pred_num', 'vs_tp']
all_res = pd.DataFrame([], columns=columns_name)

def read_xyz_file(filepath):
    """Read .xyz file and extract only the first 3 columns (X, Y, Z)."""
    data = np.loadtxt(filepath, delimiter=' ')  # Load entire file
    return data[:, :3]  # Keep only the first 3 columns

def read_obj_file(filepath):
    """Extract vertex points (v x y z) from an .obj file."""
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Extract only vertex lines
                parts = line.strip().split()
                if len(parts) >= 4:  # Ensure there are at least x, y, z values
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices) if vertices else np.empty((0, 3))  # Return empty array if no vertices

for roof_id in roof_ids:
    gt_file = os.path.join(GT_FOLDER, roof_id + GT_END)
    pred_file = os.path.join(PRED_FOLDER, roof_id + PRED_END)

    if not os.path.exists(gt_file):
        print(f"Skipping {roof_id}: Ground truth file not found.")
        continue

    vs_pred = read_xyz_file(pred_file)  # shape: (m, 3)
    vs_gt = read_obj_file(gt_file)  # shape: (n, 3)

    """ Debugging: Check shapes """
    print(f"{roof_id} -> vs_pred shape: {vs_pred.shape}, vs_gt shape: {vs_gt.shape}")

    """Step 1: Find corresponding predicted points for each ground truth point"""
    v_gt_pred_dist = eval_func.find_v_gt_pred_dist(vs_gt, vs_pred)

    """Step 2: Compute True Positives, total GT & Predicted points"""
    vs_tp, vs_tp_num, vs_gt_num, vs_pred_num = eval_func.stats_tp_gt_pred(v_gt_pred_dist, vs_gt, vs_pred, thrs_true=1)

    """Step 3: Precision and Recall"""
    precision, recall = eval_func.calc_p_r(vs_tp_num, vs_gt_num, vs_pred_num)

    """Step 4: VD(x), VD(y), VD(z)"""
    vd_x, vd_y, vd_z = eval_func.calc_vd_xyz(vs_tp)

    """Step 5: Store results"""
    res_i = [roof_id, precision, recall, vd_x, vd_y, vd_z, vs_tp_num, vs_gt_num, vs_pred_num, vs_tp]
    res_i = pd.DataFrame([res_i], columns=columns_name)
    all_res = pd.concat([all_res, res_i], ignore_index=True)

"""Calculate overall evaluation"""
vs_tp_num_all = np.sum(all_res['vs_tp_num'])
vs_gt_num_all = np.sum(all_res['vs_gt_num'])
vs_pred_num_all = np.sum(all_res['vs_pred_num'])

precision_overall, recall_overall = eval_func.calc_p_r(vs_tp_num_all, vs_gt_num_all, vs_pred_num_all)
vs_tp_all = np.vstack(all_res['vs_tp'].tolist()) if len(all_res['vs_tp']) > 0 else np.empty((0, 3))
vd_x_overall, vd_y_overall, vd_z_overall = eval_func.calc_vd_xyz(vs_tp_all)

"""Save results"""
res_overall = ['all', precision_overall, recall_overall, vd_x_overall, vd_y_overall, vd_z_overall,
               vs_tp_num_all, vs_gt_num_all, vs_pred_num_all, []]
res_overall = pd.DataFrame([res_overall], columns=columns_name)
all_res = pd.concat([all_res, res_overall], ignore_index=True)

if isSave_2_csv:
    os.makedirs(SAVE_FOLDER, exist_ok=True)  # Ensure folder exists
    all_res.to_csv(save_name, index=False)

print("Precision:\t", precision_overall)
print("Recall:\t", recall_overall)
print("VDxyz:\t", [vd_x_overall, vd_y_overall, vd_z_overall])





