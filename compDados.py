from pypcd import pypcd
import laspy
import pandas as pd
import numpy as np
import math
from pathlib import Path
import subprocess

from pypcd.pypcd import pandas_to_pypcd

#inFileLas = laspy.read('chip/chipLas/outC_Level1_61..las')  # File(path, mode="r")

#print(inFileLas.X)

#files = list(Path(".").glob("Output/marsb[1]_[5]_*.pcd"))
#files = list(Path(".").glob("Output/Scene_[1-9]_sample.pcd"))
files = list(Path(".").glob("SoftGroup/marsb/marsb*.pcd"))

for file in files:

    print(f'---------- {file.stem} ----------')

    cloud = pypcd.PointCloud.from_path(f'SoftGroup/marsb/{file.stem}.pcd')
    #cloud = pypcd.PointCloud.from_path('marsb/marsb1_1.pcd')
    dadosTodos = pd.DataFrame(np.array(cloud.pc_data))
    dadosTodos['semantic_pred'] = dadosTodos['classification']

    #print(dadosTodos)
    #dadosTodos['classification'] = dadosTodos['instance_pred']//1000

    #dadosTodos['copia_instance_pred'] = dadosTodos['instance_pred']
    #dadosTodos['copia_instance_pred'].replace(to_replace=-1, value=100, inplace=True)

    dadosTodos['classification'] = (dadosTodos['instance_pred_label'])


    #dadosTodos['classification'] = (dadosTodos['instance_pred']-dadosTodos['instance_pred_id'])/1000

    #print(dadosTodos)

    #dadosTodos.loc[(dadosTodos["instance_pred"] == -1), ['classification']] = 100
    

    #dadosTodos = dadosTodos.mask(dadosTodos['classification'] == -1, 100)

    #print(dadosTodos.loc[dadosTodos["classification"] == -1])

    #dadosTodos['classification'].replace(to_replace=-1, value=100, inplace=True)

    list_classification = np.unique(list(dadosTodos['classification']))

    print(f'Lista: {list_classification}')

    #print(dadosTodos.loc[dadosTodos["classification"] == 23])
    #print(dadosTodos.loc[(dadosTodos["instance_pred"] >= 8000) & (dadosTodos["instance_pred"] < 9000)])


    #print("______________________")
    #print(dadosTodos)
    pc = pandas_to_pypcd(dadosTodos)
    pc.save_pcd(f'SoftGroup/marsb/{file.stem}.pcd')



#----------LENDO PCD PÓS SOFTGROUP:-----------------

# cloud = pypcd.PointCloud.from_path('marsb1_1.pcd')
# # # print(cloud.pc_data)

# # # dados = pd.DataFrame()

# # # dados['x'] = np.array(cloud.pc_data['x'])

# dadosTodos = pd.DataFrame(np.array(cloud.pc_data))
# print(dadosTodos)
# dadosTodos['semantic_pred'] = dadosTodos['classification']

# print(dadosTodos)

# dadosTodos['classification'] = abs(round(dadosTodos['instance_pred'] / 1000))

# print(dadosTodos)

# pc = pandas_to_pypcd(dadosTodos)
# pc.save_pcd("marsb1_1.pcd")

# dadosTodos = dadosTodos.head(1000)
# dadosTodos.to_excel("outputpcdPdal.xlsx")

# cloud1 = pypcd.PointCloud.from_path('potree_10_1.pcd')
# cloud2 = pypcd.PointCloud.from_path('potree_10_2.pcd')

# dados = pd.DataFrame(np.array(cloud1.pc_data))

# dados2 = pd.DataFrame(np.array(cloud2.pc_data))

# dadosFull = pd.DataFrame([np.array(cloud1.pc_data), np.array(cloud2.pc_data)])

# print(dadosFull.head(10))

# dados3 = dados.append(np.array(cloud2.pc_data), ignore_index=True)
# output_cols = [
#             "x",
#             "y",
#             "z",
#             "rgb",
#             "Intensity",
#         ]
# pc = pandas_to_pypcd(dadosFull[output_cols])
        
# pc.save_pcd("teste_10_full.pcd")

files = list(Path(".").glob("Output/moerdijk_F_semantic/moerdijk*.las"))

for file in files:
    inFileLas = laspy.read(f'Output/moerdijk_F_semantic/{file.stem}.las')  # File(path, mode="r")
    #inFileLas.add_extra_dim(laspy.ExtraBytesParams(name="classification", type=np.uint64))
    print(inFileLas.instance_pred_label)
    #inFileLas.classification = inFileLas.instance_pred_label

    setattr(inFileLas, 'classification', inFileLas.instance_pred_label)

    print(inFileLas.classification)
    
    
    print(list(inFileLas.point_format.extra_dimension_names))
    inFileLas.write(f"Output/moerdijk_F_semantic/{file.stem}_instance.las")


#---
# inFileLasM = laspy.read('teste/marsb.las')  # File(path, mode="r")

# inFileLasF = laspy.read('teste/fluminense.las')  # File(path, mode="r")

# dadosM = pd.DataFrame()
# dadosF = pd.DataFrame()

# # dadosTodos2['x_Las'] = np.array(inFileLas.X).T
# # dadosTodos2['y_Las'] = np.array(inFileLas.Y).T
# # dadosTodos2['z_Las'] = np.array(inFileLas.Z).T


# dadosM['red'] = np.array(inFileLasM.red).T
# dadosM['green'] = np.array(inFileLasM.green).T
# dadosM['blue'] = np.array(inFileLasM.blue).T

# dadosF['red'] = np.array(inFileLasF.red).T
# dadosF['green'] = np.array(inFileLasF.green).T
# dadosF['blue'] = np.array(inFileLasF.blue).T

# print(dadosF['red'].head(5))
# print(dadosM['red'].head(5))

# print(dadosF['red'].max())

# print(dadosM['red'].max())


#df_1 = dadosTodos2.head(10000)

#print(pd.isna(dadosTodos2))
#if(pd.isna(dadosTodos2['x_Las'])):
#    print('NaN')

#print(pd.isna(dadosTodos2['x_Las'])[2])
# for num in range(1,61458146):
#     if(pd.isna(dadosTodos2['x_Las'])[num] == True):
#         print('NaN')

#dadosTodos2.to_excel("dados_5.xlsx")



#----------LENDO LAS ANTES SOFTGROUP:-----------------

# inFileLas = laspy.read('outC_Level1_61.las')  # File(path, mode="r")

# #print(inFileLas.point_format)

# print(list(inFileLas.point_format.standard_dimension_names))
# print(list(inFileLas.point_format.extra_dimension_names))


# inFileLas.add_extra_dim(laspy.ExtraBytesParams(
#     name="codification",
#     type=np.uint64
# ))

# inFileLas.add_extra_dim(laspy.ExtraBytesParams(name="mysterious", type="3f8"))
# # inFileLas = laspy.read('testeoutC_Level1_61..las')  # File(path, mode="r")


# #print(inFileLas.X)
# #print(inFileLas.classification)
# inFileLas.Y
# inFileLas.Z

# cloud = pypcd.PointCloud.from_path('v1-E_decode/outC_Level1_61..pcd')
#print(cloud.pc_data["x"])
#print(cloud.pc_data["semantic_pred"])


# x_Las = np.array(inFileLas.X * 0.01).T
# y_Las = np.array(inFileLas.Y * 0.01).T
# z_Las = np.array(inFileLas.Z * 0.01).T
# classification_Las = np.array(inFileLas.classification).T

# semantic_pred_Pcd = np.array(cloud.pc_data["semantic_pred"]).T
# x_Pcd = np.array(cloud.pc_data["x"]).T
# y_Pcd = np.array(cloud.pc_data["y"]).T
# z_Pcd = np.array(cloud.pc_data["z"]).T


# df = pd.DataFrame()

# df['x_Las'] = np.array(inFileLas.X* 0.01).T
# df['y_Las'] = np.array(inFileLas.Y* 0.01).T
# df['z_Las'] = np.array(inFileLas.Z* 0.01).T
# df['classification_Las'] = np.array(inFileLas.classification).T

# df['semantic_pred_Pcd'] = np.array(cloud.pc_data["semantic_pred"]).T
# df['x_Pcd'] = np.array(cloud.pc_data["x"]).T
# df['y_Pcd'] = np.array(cloud.pc_data["y"]).T
# df['z_Pcd'] = np.array(cloud.pc_data["z"]).T

# #df_1 = df.head(1000000)

# #df_1.to_excel("output.xlsx")


# if(len(z_Las) == len(z_Pcd)):
#     print("Iguais!")

# cont = 0

# # for n in range(0,len(z_Las)):
# #     if(z_Las[n] != z_Pcd[n]):
# #         cont += 1

# print(z_Las[500963])
# print(z_Pcd[500963])

# soma_x_Las = df['x_Las'].sum()
# media_x_Las = soma_x_Las/df.shape[0]

# soma_x_Pcd = df['x_Pcd'].sum()
# media_x_Pcd = soma_x_Pcd/df.shape[0]

# df['dist_centro_x_Las'] = df['x_Las'] - media_x_Las
# df['dist_centro_x_Pcd'] = df['x_Pcd'] - media_x_Pcd


# soma_y_Las = df['y_Las'].sum()
# media_y_Las = soma_y_Las/df.shape[0]

# soma_y_Pcd = df['y_Pcd'].sum()
# media_y_Pcd = soma_y_Pcd/df.shape[0]

# df['dist_centro_y_Las'] = df['y_Las'] - media_y_Las
# df['dist_centro_y_Pcd'] = df['y_Pcd'] - media_y_Pcd


# df_1 = df.head(1000)

# df_1.to_excel("output2.xlsx")