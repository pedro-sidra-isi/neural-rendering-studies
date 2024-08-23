# %%
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d


# Função para marcar ruídos e verificar proximidade
# Pa: nuvem de pontos alvo
# Pr: nuvem de pontos de referência
# max_dist: distância máxima para considerar um ponto como válido
def marcar_ruido_e_proximidade(Pa, Pr, max_dist=0.1):

    # Criação de árvores KD para ambas as nuvens de pontos
    arvore_R = KDTree(Pr)

    # Consulta dos pontos de Pa na árvore de Pr para obter as distâncias
    distancias, _ = arvore_R.query(Pa)

    # Identificação dos índices dos pontos de ruído (distâncias maiores ou iguais a max_dist)
    indices_ruido = np.where(distancias >= max_dist)[0]

    # Identificação dos índices e distâncias dos pontos válidos (distâncias menores que max_dist)
    indices_validos = np.where(distancias < max_dist)[0]
    dist_validas = distancias[indices_validos]

    # Retorno dos índices de ruído, índices de distâncias válidas e as distâncias válidas
    return indices_ruido, indices_validos, dist_validas


# %% Carregamento das nuvens de pontos dos arquivos .ply
Pa = o3d.io.read_point_cloud("/mnt/d/data/teste.ply")
Pr = o3d.io.read_point_cloud("/mnt/d/data/reference_v1e2_subsample.ply")

# %% Conversão das nuvens de pontos para arrays numpy
Pa_np = np.asarray(Pa.points)
Pr_np = np.asarray(Pr.points)

# %% Execução da função de marcação de ruídos e proximidade
indices_ruido, indices_validos, distancias_validas = marcar_ruido_e_proximidade(
    Pa_np, Pr_np, max_dist=0.1
)

# Impressão dos resultados
print("Pontos de ruído em Pa :", indices_ruido)
print("Pontos válidos em Pa :", indices_validos)
print("Distâncias dos pontos válidos :", distancias_validas)


# %%
import plotly.express as px

fig = px.histogram(
    x=valid_distances, nbins=100000, labels={"x": "Dist"}, title=" dist Pontos"
)
fig.show()


# %%


total_points = np.asarray(Pa.points).shape[0]
captured_points = valid_distances.shape[0]

surface_captured = (captured_points / total_points) * 100

average_distance = valid_distances.mean()


variance_of_error = valid_distances.var()

print(f"Porcentagem da Superfície(+-) : {surface_captured}%")
print(f"Distância Média : {average_distance}")
print(f"Variância: {variance_of_error}")
# %%
# from scipy.spatial import Delaunay
# import numpy as np

# def estimate_covered_area(points):

#     tri = Delaunay(points[:, :2])
#     triangles = points[tri.simplices]

#     area = 0.0
#     for triangle in triangles:
#         area += 0.5 * np.abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))

#     area = area.sum()

#     return area


# Pa_points = Pa_np[valid_indices]
# Pr_points = Pr_np

# covered_area = estimate_covered_area(Pa_points)
# total_area = estimate_covered_area(Pr_points)

# print(f"Área coberta: {covered_area}")
# print(f"Área total: {total_area}")


# surface_captured = (covered_area / total_area) * 100
# print(f"Porcentagem da Superfície Capturada: {surface_captured}%")


# %%
from skimage import measure
import plotly.graph_objects as go
import numpy as np


def create_and_plot_mesh(points, level=None, grid_size=30):
    # Define the volume grid for the scalar field
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    x, y, z = (
        np.linspace(min_bounds[0], max_bounds[0], grid_size),
        np.linspace(min_bounds[1], max_bounds[1], grid_size),
        np.linspace(min_bounds[2], max_bounds[2], grid_size),
    )
    X, Y, Z = np.meshgrid(x, y, z)
    grid = np.full(X.shape, np.inf)

    # Calculate the minimum distance from each grid point to any point
    for point in points:
        grid = np.minimum(
            grid,
            np.sqrt((X - point[0]) ** 2 + (Y - point[1]) ** 2 + (Z - point[2]) ** 2),
        )

    # Automatically determine an appropriate level if not provided
    if level is None:
        level = np.mean(grid)

    # Use the marching cubes algorithm to create the mesh
    verts, faces, normals, _ = measure.marching_cubes(grid, level)
    # Plotting with plotly
    mesh = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="blue",
        opacity=0.5,
    )
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"),
        title="3D Mesh from Valid Points",
    )
    fig.show()


# Use this function after you have the valid_points and valid_distances
valid_points = Pa_np[valid_indices]  # Assumed you have this from your previous code
create_and_plot_mesh(
    valid_points, level=np.percentile(valid_distances, 50)
)  # Using 50th percentile as an example level


# %%
##TESTESEEEEEE

import numpy as np
from scipy.spatial import KDTree
import open3d as o3d


# Função para calcular as distâncias entre nuvens de pontos e estatísticas
def calcular_estatisticas_distancias(Pa, Pr, max_dist=0.1):

    # Criação de árvores KD para ambas as nuvens de pontos
    arvore_R = KDTree(Pr)

    # Consulta dos pontos de Pa na árvore de Pr para obter as distâncias
    distancias_A_para_R, _ = arvore_R.query(Pa)

    # Cálculo dos valores estatísticos
    min_dist = np.min(distancias_A_para_R)
    max_dist = np.max(distancias_A_para_R)
    avg_dist = np.mean(distancias_A_para_R)
    sigma = np.std(distancias_A_para_R)

    # Consulta dos pontos de Pr na árvore de Pa para obter as distâncias de Pr para Pa
    arvore_A = KDTree(Pa)
    distancias_R_para_A, _ = arvore_A.query(Pr)

    # Cálculo do erro máximo das distâncias de Pr para Pa
    max_error = np.max(distancias_R_para_A)

    # Cálculo da porcentagem de cobertura usando Voxel Grid
    voxel_size = max_dist  # Pode ajustar o tamanho do voxel conforme necessário
    voxel_grid_Pa = o3d.geometry.VoxelGrid.create_from_point_cloud(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pa)), voxel_size=voxel_size
    )
    voxel_grid_Pr = o3d.geometry.VoxelGrid.create_from_point_cloud(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pr)), voxel_size=voxel_size
    )

    # Obter voxels ocupados por cada nuvem
    voxels_Pa = set(tuple(voxel.grid_index) for voxel in voxel_grid_Pa.get_voxels())
    voxels_Pr = set(tuple(voxel.grid_index) for voxel in voxel_grid_Pr.get_voxels())

    print(len(voxels_Pa))
    print("aaa")
    print(len(voxels_Pr))
    # Calcular a interseção de voxels para obter a porcentagem de cobertura
    voxels_intersecao = voxels_Pa.intersection(voxels_Pr)
    porcentagem_cobertura = len(voxels_intersecao) / len(voxels_Pr) * 100

    # Retorno dos valores calculados
    return min_dist, max_dist, avg_dist, sigma, max_error, porcentagem_cobertura


Pa = o3d.io.read_point_cloud("/mnt/d/data/E4.las.subsampled_mesh3.ply")
Pr = o3d.io.read_point_cloud("/mnt/d/data/reference_v1e2_subsample.ply")


Pa_np = np.asarray(Pa.points)
Pr_np = np.asarray(Pr.points)

min_dist, max_dist, avg_dist, sigma, max_error, porcentagem_cobertura = (
    calcular_estatisticas_distancias(Pa_np, Pr_np, max_dist=0.1)
)

# Impressão dos resultados
print("Distância mínima:", min_dist)
print("Distância máxima:", max_dist)
print("Distância média:", avg_dist)
print("Desvio padrão (sigma):", sigma)
print("Erro máximo (Pr para Pa):", max_error)
print("Porcentagem de cobertura de Pr por Pa:", porcentagem_cobertura)


# %%
