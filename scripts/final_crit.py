# %%
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d


def marcar_ruido_e_proximidade(Pa, Pr, max_dist=0.1):

    # nparay
    Pa_np = np.asarray(Pa.points)
    Pr_np = np.asarray(Pr.points)

    arv_R = KDTree(Pr_np)
    arv_A = KDTree(Pa_np)

    # distâncias pa
    dists, _ = arv_R.query(Pa_np)

    # ruído (distancias >= max dist)
    idx_ruido = np.where(dists >= max_dist)[0]

    # pontos validos
    idx_validos = np.where(dists < max_dist)[0]
    dists_validas = dists[idx_validos]

    idx_oclusos = []

    # pontos oclusos na nuvem de referência
    for j, Pr_j in enumerate(Pr_np):
        _, idx_vizinho_A = arv_A.query(Pr_j)
        Pa_k = Pa_np[idx_vizinho_A]
        dist = np.linalg.norm(Pa_k - Pr_j)
        if dist > max_dist:
            idx_oclusos.append(j)

    # porcentagem de pontos bem representados
    total_ref = len(Pr_np)
    total_oclusos = len(idx_oclusos)

    if total_ref > 0:
        pct_representada = 100 * (1 - total_oclusos / total_ref)
    else:
        pct_representada = 0  # caso nao tenha pontos

    if len(dists) > 0:
        max_distancia = np.max(dists)
        min_distancia = np.min(dists)
        avg_distancia = np.mean(dists)
        sigma_distancia = np.std(dists)
    else:
        max_distancia = min_distancia = avg_distancia = sigma_distancia = 0

    return (
        idx_ruido,
        idx_validos,
        dists_validas,
        idx_oclusos,
        pct_representada,
        max_distancia,
        min_distancia,
        avg_distancia,
        sigma_distancia,
        total_ref,
    )


# %% ANALISE GERAL DA NUVEM
Pa = o3d.io.read_point_cloud("/mnt/d/data/maquetes/ptsam_voxelNERF.ply")
Pr = o3d.io.read_point_cloud("/mnt/d/data/maquetes/reference_v1e2_subsample.ply")

print("Nuvens de pontos carregadas.")


(
    idx_ruido,
    idx_validos,
    dists_validas,
    idx_oclusos,
    pct_representada,
    max_distancia,
    min_distancia,
    avg_distancia,
    sigma_distancia,
    total_ref,
) = marcar_ruido_e_proximidade(Pa, Pr)


print(f"Distância média: {avg_distancia}")
print(f"Desvio padrão (sigma): {sigma_distancia}")
print(f"Número de índices de ruído: {len(idx_ruido)}")
print(f"Número de índices válidos: {len(idx_validos)}")
print(f"Número total de pontos na nuvem avalaida: {len(idx_validos)+len(idx_ruido)}")
print(f"Número de índices de pontos oclusos: {len(idx_oclusos)}")
print(f"Porcentagem da nuvem de referência bem representada: {pct_representada:.2f}%")
print(f"Total de pontos na nuvem de referência: {total_ref}")

# %% COMPARATIVO


import pandas as pd


pa_files = [
    # "/mnt/d/data/maquetes/ptsam_PGM.ply",
    #  "/mnt/d/data/maquetes/ptsam_NERF.ply",
    # "/mnt/d/data/maquetes/ptsam_3GDS.ply",
    #  "/mnt/d/data/maquetes/ptsam_voxelNERF.ply",
    # "/mnt/d/data/maquetes/ptsam_voxelPedro.ply",
    # "/mnt/d/data/maq_0407_sam/pro/point_cloud_gauss.ply",
    "/mnt/d/data/artemis/0207_pro/ad_point_cloud.ply",
]


segmentadas_paths = [
    "/mnt/d/data/maquetes/maq_seg/Valvulas.ply",
    "/mnt/d/data/maquetes/maq_seg/Canos.ply",
    "/mnt/d/data/maquetes/maq_seg/componentes_gerais_tub.ply",
    "/mnt/d/data/maquetes/maq_seg/Estrutura.ply",
    "/mnt/d/data/maquetes/maq_seg/flanges_cg.ply",
    "/mnt/d/data/maquetes/maq_seg/Motores_tanques.ply",
    "/mnt/d/data/maquetes/maq_seg/Superficie.ply",
]

resultados = []

for pa_file in pa_files:
    Pa = o3d.io.read_point_cloud(pa_file)

    for path in segmentadas_paths:
        Pr_segmentada = o3d.io.read_point_cloud(path)

        (
            idx_ruido,
            idx_validos,
            dists_validas,
            idx_oclusos,
            pct_representada,
            max_distancia,
            min_distancia,
            avg_distancia,
            sigma_distancia,
            total_ref,
        ) = marcar_ruido_e_proximidade(Pa, Pr_segmentada)

        resultado = {
            "Nome do Arquivo Pa": pa_file.split("/")[-1],
            "Nome do Arquivo Pr": path.split("/")[-1],
            "Distância Média": avg_distancia,
            "Desvio Padrão (sigma)": sigma_distancia,
            "Porcentagem Representada": pct_representada,
        }

        resultados.append(resultado)

df_resultados = pd.DataFrame(resultados)


# %%

import plotly.express as px


df_resultados["Nome do Arquivo Pa"] = (
    df_resultados["Nome do Arquivo Pa"]
    .str.replace("ptsam_", "")
    .str.replace(".ply", "")
)
df_resultados["Nome do Arquivo Pr"] = df_resultados["Nome do Arquivo Pr"].str.replace(
    ".ply", ""
)

template = "plotly_white"
color_discrete_sequence = px.colors.qualitative.Set2

# Dist
fig1 = px.line(
    df_resultados,
    x="Nome do Arquivo Pr",
    y="Distância Média",
    color="Nome do Arquivo Pa",
    title="Distância Média por Parte e Grupo",
    template=template,
    color_discrete_sequence=color_discrete_sequence,
    markers=True,
)

fig1.update_layout(
    title_font_size=20, xaxis_title="Partes", yaxis_title="Distância Média"
)

# Sigma
fig2 = px.line(
    df_resultados,
    x="Nome do Arquivo Pr",
    y="Desvio Padrão (sigma)",
    color="Nome do Arquivo Pa",
    title="Desvio Padrão por Parte e Grupo",
    template=template,
    color_discrete_sequence=color_discrete_sequence,
    markers=True,
)

fig2.update_layout(
    title_font_size=20, xaxis_title="Partes", yaxis_title="Desvio Padrão (sigma)"
)


fig3 = px.line(
    df_resultados,
    x="Nome do Arquivo Pr",
    y="Porcentagem Representada",
    color="Nome do Arquivo Pa",
    title="Porcentagem Representada por Segmentação",
    template=template,
    color_discrete_sequence=color_discrete_sequence,
    markers=True,
)

fig3.update_layout(
    title_font_size=20, xaxis_title="Partes", yaxis_title="Porcentagem Representada (%)"
)


fig1.show()
fig2.show()
fig3.show()


# %%
# Box plot com média e desvio padrão
import matplotlib.pyplot as plt

# Preparar os dados
df_resultados["Nome do Arquivo Pa"] = (
    df_resultados["Nome do Arquivo Pa"]
    .str.replace("ptsam_", "")
    .str.replace(".ply", "")
)

# Calculando as métricas
mean_values = df_resultados.groupby("Nome do Arquivo Pa")[
    "Porcentagem Representada"
].mean()
std_values = df_resultados.groupby("Nome do Arquivo Pa")[
    "Porcentagem Representada"
].std()

# Plotando o box plot com média e desvio padrão
fig, ax = plt.subplots(figsize=(12, 8))

# Box plot
df_resultados.boxplot(
    column="Porcentagem Representada", by="Nome do Arquivo Pa", ax=ax, grid=False
)

# Adicionando médias e desvios padrões
for i, nome_pa in enumerate(mean_values.index):
    mean = mean_values[nome_pa]
    std = std_values[nome_pa]
    ax.plot(i + 1, mean, "ro")
    ax.errorbar(i + 1, mean, yerr=std, fmt="o", color="red", capsize=5)

# Customizando o gráfico
plt.title("Box Plot com Média e Desvio Padrão das Porcentagens Representadas")
plt.suptitle("")
plt.xlabel("Nome do Arquivo Pa")
plt.ylabel("Porcentagem Representada")
plt.grid(True)

# Exibindo o gráfico
plt.show()
