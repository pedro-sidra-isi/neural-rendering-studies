#%%
import numpy as np
import open3d as o3d



pcd = o3d.io.read_point_cloud("/home/guto/Projetos/fotogrametria/dist/data/E4.las.subsampled.ply")
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_plotly([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


# %%
