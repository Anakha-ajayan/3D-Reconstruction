# 3D-Wireframe reconstruction of building roofs from airborne LiDAR point cloud.
This study presents an integrated approach
for reconstructing wireframe models of building roofs segmented from airborne
LiDAR point cloud data. Using the DALES dataset, we applied a sophisticated
density-based clustering algorithm (DBSCAN) to segment disjoint buildings. Postprocessing
techniques were employed to generate consistent and clean clustered
building roof point clouds by applying geometric normalization and noise removal
before feeding them into the network. A rule-based method was implemented
to identify potential roof vertices and structural lines, aiding in the extraction of
key architectural features. To enhance both performance and interpretability, a
binary edge prediction framework was established for wireframe generation, employing
ensemble machine learning models that leverage engineered geometric
features and SHAP-based feature selection. The proposed model was assessed
using publicly available international urban datasets and specifically evaluated on
the Trivandrum Aerial LiDAR Dataset (TALD), a regional real-time dataset of Thiruvanathapuram, Kerala.



Order of Execution :
1. segment.ipynb

Purpose: Segment individual building point clouds from a semantically labeled scene point cloud.

Note: The input data required for segmentation is not included in this repository due to privacy policies.

2. roofVe_main--testing.ipynb

Purpose: Obtain the corner vertices of the segmented building roofs.

3. binary_edge_prediction.ipynb

Purpose: Predict the edges based on corner vertices and reconstruct wireframe models of building roofs.


Acknowledgements :

This work is built upon RoofVE. We sincerely thank the authors for their outstanding work and for making their repository available.
We also thank the Indian Institute of Space Science and Technology (IIST) for providing the TALD dataset and their continuous support throughout this project.
