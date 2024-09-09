#tutsteste
# Carregar funções

import plotly.express as px
import pandas as pd
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree

# Matthews Correlation Coefficient -  qualidade de uma classificação binária
def calcular_mcc(tp, tn, fp, fn):
    numerador = (tp * tn) - (fp * fn)
    denominador = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominador == 0:
        return 0 
    else:
        return numerador / denominador


# Função para segmentar Pa por proximidade das partes de Pr
def segmentar_pa_por_distancia(Pa, Pr_segmentadas):
    Pa_np = np.asarray(Pa.points)

    labels = np.zeros(Pa_np.shape[0], dtype=int)
    min_dists = np.full(Pa_np.shape[0], np.inf)

    # Iterar sobre cada parte segmentada de Pr
    for i, Pr_segmentada in enumerate(Pr_segmentadas):
        Pr_np = np.asarray(Pr_segmentada.points)
        arv_R = KDTree(Pr_np)
        dists_RA, _ = arv_R.query(Pa_np)

        # Atualizar os rótulos e as distâncias mínimas
        mask = dists_RA < min_dists
        labels[mask] = i + 1
        min_dists[mask] = dists_RA[mask]

    df_pa = pd.DataFrame(Pa_np, columns=["x", "y", "z"])
    df_pa["label"] = labels
    df_pa["min_dist"] = min_dists

    return df_pa


# Função para avaliar a correspondência de Pa com as partes segmentadas de Pr usando os labels
def avaliar_correspondencia_segmentada(df_pa, Pr_segmentadas, max_dist):
    resultados = []

    for i, Pr_segmentada in enumerate(Pr_segmentadas):
        Pr_np = np.asarray(Pr_segmentada.points)

        # Selecionar pontos de Pa que pertencem ao segmento atual
        Pa_filtrada = df_pa[df_pa["label"] == (i + 1)]
        Pa_filtrada_np = Pa_filtrada[["x", "y", "z"]].to_numpy()

        # Árvore pra Pr 
        arv_R = KDTree(Pr_np)
        dists_RA, _ = arv_R.query(Pa_filtrada_np)  # Pa -> Pr

        # Quantidade de pontos de A que têm um ponto em R_i dentro de D_max
        total_pa = len(Pa_filtrada_np) # (True positives + False positives)
        dists_validos_RA = dists_RA[dists_RA < max_dist]
        total_correspondencias = len(dists_validos_RA)

        # Precision = | { A_i | DNN(A_i, R_i) < D_max } | / | A |
        if total_pa > 0:
            precisao = (total_correspondencias / total_pa)
        else:
            precisao = 0

        # Construir a árvore para P_a segmentadaa para o cálculo do recall
        arv_A = KDTree(Pa_filtrada_np)
        dists_AR, _ = arv_A.query(Pr_np)  # Pr -> Pa

        # Quantidade de pontos de R_i que têm um ponto em A dentro de D_max
        total_ref = len(Pr_np) # (True positives + False Negatives)
        pontos_bem_representados = np.sum(dists_AR < max_dist)

        # Recall = | { R_i | DNN(R_i, A) < D_max } | / | R_i |
        if total_ref > 0:
            recall = (pontos_bem_representados / total_ref) 
        else:
            recall = 0 

        # F1-Score
        if precisao + recall > 0:
            f1_score = 2 * (precisao * recall) / (precisao + recall)
        else:
            f1_score = 0

        false_positives = total_pa - total_correspondencias  # Pontos de Pa incorretamente associados
        false_negatives = total_ref - pontos_bem_representados  # Pontos de Pr não representados
        true_positives = pontos_bem_representados
        true_negatives = total_ref - false_negatives  # Aproximação: o que resta de Pr corretamente associado

        # Calcular o MCC
        mcc = calcular_mcc(true_positives, true_negatives, false_positives, false_negatives)


        # Estatísticas Pa para Pr
        if len(dists_RA) > 0:
            max_distancia = np.max(dists_RA)
            min_distancia = np.min(dists_RA)
            avg_distancia = np.mean(dists_RA)
            sigma_distancia = np.std(dists_RA)
        else:
            max_distancia = min_distancia = avg_distancia = sigma_distancia = 0

        resultados.append(
            {
                "Segmento": f"Parte {i + 1}",
                "Distância Média": avg_distancia,
                "Desvio Padrão (sigma)": sigma_distancia,
                "Distância Máxima": max_distancia,
                "Distância Mínima": min_distancia,
                "Recall": recall,
                "Precisão": precisao,
                "F1-Score": f1_score,
                "MCC": mcc,
            }
        )

    df_resultados = pd.DataFrame(resultados)

    return df_resultados



def plotar_graficos(df_resultados):
    template = "plotly_white"
    color_discrete_sequence = px.colors.qualitative.Set2

    fig1 = px.line(
        df_resultados,
        x="Segmento",
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
    fig1.show()

    fig2 = px.line(
        df_resultados,
        x="Segmento",
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
    fig2.show()

    df_resultados['Recall (%)'] = df_resultados['Recall'] * 100
    fig3 = px.line(
        df_resultados,
        x="Segmento",
        y="Recall (%)",
        color="Nome do Arquivo Pa",
        title="Recall por Segmentação",
        template=template,
        color_discrete_sequence=color_discrete_sequence,
        markers=True,
    )
    fig3.update_layout(
        title_font_size=20,
        xaxis_title="Partes",
        yaxis_title="Recall (%)",
    )
    fig3.show()

    df_resultados['Precisão (%)'] = df_resultados['Precisão'] * 100
    fig4 = px.line(
        df_resultados,
        x="Segmento",
        y="Precisão",
        color="Nome do Arquivo Pa",
        title="Precisão por Segmentação",
        template=template,
        color_discrete_sequence=color_discrete_sequence,
        markers=True,
    )
    fig4.update_layout(
        title_font_size=20,
        xaxis_title="Partes",
        yaxis_title="Precisão",
    )
    fig4.show()


# Função para processar segmentação e avaliação
def processar_segmentacao_e_avaliacao(pa_files, segmentadas_paths, max_dist):
    Pr_segmentadas = [o3d.io.read_point_cloud(path) for path in segmentadas_paths]
    resultados = []

    
    for pa_file in pa_files:
        Pa = o3d.io.read_point_cloud(pa_file)

        # Segmentar Pa
        df_pa_segmentada = segmentar_pa_por_distancia(Pa, Pr_segmentadas)

        # Correspondência para cada segmento de Pr
        df_resultado = avaliar_correspondencia_segmentada(
            df_pa_segmentada, Pr_segmentadas, max_dist
        )

    
        df_resultado["Nome do Arquivo Pa"] = pa_file.split("/")[-1]
        resultados.append(df_resultado)

    df_todos_resultados = pd.concat(resultados, ignore_index=True)

    return df_todos_resultados

pa_files = [
    "/mnt/d/data/maquetes/ptsam_PGM.ply",
    "/mnt/d/data/maquetes/ptsam_NERF.ply",
    "/mnt/d/data/maquetes/ptsam_3GDS.ply",
    "/mnt/d/data/maquetes/ptsam_voxelNERF.ply",
    "/mnt/d/data/maquetes/ptsam_voxelPedro.ply",
    "/mnt/d/data/maquetes/reference_v1e2_subsample.ply",
    "/mnt/d/data/artemis/0207_pro/ptc_ali_0207_pro.ply",
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
# %%
# Carregar as nuvens e gerar métricas
max_dist = 0.1
df_resultados = processar_segmentacao_e_avaliacao(pa_files, segmentadas_paths, max_dist)
# %%
# Plotar gráficos dos resultados
plotar_graficos(df_resultados)
# %%
print(df_resultados[['Segmento', 'Recall', 'Precisão', 'F1-Score', 'MCC']])
# %%
