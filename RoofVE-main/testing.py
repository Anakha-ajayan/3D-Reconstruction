import os
import glob
import numpy as np
import pandas as pd
import open3d as o3d

import utils.module_voxelizaton as mdl_voxel
import utils.visualization as visual
import utils.module_findStruct_v3_NoOver as mdl_fdStruct_v3_noOver
import utils.read_data as read_data

# Set paths and parameters
OPEN_FOLDER = r'C:\Users\akhil\OneDrive\Documents\point2building\RoofVE-main\Trivandrum_clusters_subtile_7'
SAVE_FOLDER = r'C:\Users\akhil\OneDrive\Documents\point2building\RoofVE-main\pred_res\trivandrum'
GRID_SIZE = 0.7# Voxelization grid size
max_search_radius_param = +1  # Search radius parameter

# Ensure save folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Get all point cloud files (.ply and .xyz)
pcd_list = glob.glob(os.path.join(OPEN_FOLDER, '*.ply')) + glob.glob(os.path.join(OPEN_FOLDER, '*.xyz'))

def load_point_cloud(file_path):
    """Load point cloud data from .ply or .xyz file."""
    if file_path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)  # Extract xyz
    elif file_path.endswith('.xyz'):
        return np.loadtxt(file_path, delimiter=' ', usecols=(0, 1, 2))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

# Iterate over test point clouds
for fi, file_path in enumerate(pcd_list):
    print(f'===================== Predicting for {fi}: {os.path.basename(file_path)} =====================')
    
    # Load point cloud file
    xyz = load_point_cloud(file_path)

    # Preprocessing: Main Direction and Voxelization
    maindir = mdl_voxel.MainDirection(xyz)
    rect = maindir.getMainDirection(img_resolution=256, pad_size=10)
    M, xyz_rot = maindir.rotatePoints(rect=rect)
    
    voxelize = mdl_voxel.Voxelization(xyz_rot, grid_size=GRID_SIZE, thres_pcnt=0)
    voxels_center, voxels_idx = voxelize.getVoxels()

    # Predict Structure
    op_struct = mdl_fdStruct_v3_noOver.FindRoofStructure()
    
    voxel_pd_topsurf = op_struct.find_topSurf(voxels_idx)
    voxel_pd_struct = op_struct.rm_innerPoint_v2(voxel_pd_topsurf, max_search_radius_param)
    voxel_cand_pd = op_struct.rm_isolateP_v2(voxel_pd_struct)
    
    voxel_cand_line_pd, line_interior_idx = op_struct.save_lineP_v2(voxel_pd_topsurf, voxel_cand_pd)

    # Process line segments and intersections
    cand_line_idx_pd = pd.DataFrame({'interior_idx': line_interior_idx}, dtype='object')
    cand_line_idx_pd[['id_z', 'l_a', 'l_b', 'l_c', 'l_len']] = cand_line_idx_pd.apply(
        lambda row: op_struct.sep_lG_and_get_lFunc(row, voxel_cand_line_pd), axis=1,
        result_type="expand")
    
    cand_line_idx_pd['id_z_bp'] = cand_line_idx_pd['id_z']
    cand_line_idx_pd.loc[cand_line_idx_pd['id_z_bp'] == 1, 'id_z'] = 0

    # Remove abnormal lines and calculate intersections
    cand_line_idx_pd = op_struct.rm_abnormal_line(cand_line_idx_pd, voxel_cand_line_pd)
    cand_line_idx_pd = op_struct.get_cand_intersect_layer(cand_line_idx_pd)
    
    cand_line_idx_pd['intersectPs'] = cand_line_idx_pd.apply(
        lambda row: op_struct.calc_intersectP(row, cand_line_idx_pd, voxel_cand_line_pd), axis=1)
    
    cand_line_idx_pd['new_lines'] = cand_line_idx_pd.apply(
        lambda row: op_struct.split_intersected_line(row, voxel_cand_line_pd), axis=1)

    # Extract the final corner points
    LC_line_idx_pd = op_struct.divide_ngbr_sl_clusters(cand_line_idx_pd)
    fin_line_idx_pd = op_struct.extract_best_sl(LC_line_idx_pd, voxel_cand_line_pd)
    fin_line_idx_pd = fin_line_idx_pd[fin_line_idx_pd['isSave'] == True]
    
    # Flatten `new_lines` safely
    all_lines = [x for sublist in fin_line_idx_pd['new_lines'].dropna() for x in (sublist if isinstance(sublist, list) else [sublist])]
    all_lines_flat = sum([x if isinstance(x, list) else [x] for x in all_lines], [])
    fin_corner_idx = np.unique(all_lines_flat).tolist()

    
    fin_corner_idx_pd = voxel_cand_line_pd[voxel_cand_line_pd['ofid'].isin(fin_corner_idx)]

    # Postprocessing: Re-rotate to geo-coordinates
    op_geo_rerot = mdl_fdStruct_v3_noOver.FindCorners(fin_corner_idx_pd, voxels_center, M)
    fin_geo_corners = op_geo_rerot.get_Geo_Corners()

    # Save Results
    save_name = os.path.join(SAVE_FOLDER, os.path.basename(file_path).rsplit('.', 1)[0])
    np.savetxt(save_name + '_finCor_Geo.txt', fin_geo_corners, delimiter=",", fmt="%.8f")

    # Save other intermediate results
    read_data.save_dataframe(save_name + '_0topsurf.csv', voxel_pd_topsurf)
    read_data.save_dataframe(save_name + '_1struc.csv', voxel_pd_struct)
    read_data.save_dataframe(save_name + '_2cand.csv', voxel_cand_pd)
    read_data.save_dataframe(save_name + '_3structL.csv', voxel_cand_line_pd)
    read_data.save_dataframe(save_name + '_3-1finLine_idx.csv', fin_line_idx_pd)
    
    voxels_fin_line_idx = np.unique(sum(fin_line_idx_pd['interior_idx'].dropna().tolist(), [])).tolist()
    voxels_fin_line_pd = voxel_cand_line_pd[voxel_cand_line_pd['ofid'].isin(voxels_fin_line_idx)]
    read_data.save_dataframe(save_name + '_3-1finLine.csv', voxels_fin_line_pd)
    read_data.save_dataframe(save_name + '_4finCor.csv', fin_corner_idx_pd)

print("Predictions for all test samples are saved.")








