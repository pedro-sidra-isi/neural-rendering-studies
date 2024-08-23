
#%%
from PIL import Image
import numpy as np
import open3d as o3d
Image.MAX_IMAGE_PIXELS = None


image_path = '/mnt/d/data/maq_0407_sam/single/images.density_slices_608x416x608.png'


image = Image.open(image_path)
image_array = np.array(image)


slice_largura = 608
slice_altura = 416
slices = 608

# dim da img
im_largura, im_altura = image.size

# slices por linha e coluna
slices_linha = im_largura // slice_largura
slices_coluna = im_altura // slice_altura


voxels = []

for slice_idx in range(slices):
    # posição da slice na imagem
    lin = slice_idx // slices_linha
    col = slice_idx % slices_linha
    x_offset = col * slice_largura
    y_offset = lin * slice_altura

    # pixels da slice
    for y in range(slice_altura):
        for x in range(slice_largura):
            # densidade do pixel
            density = image_array[y + y_offset, x + x_offset]
            
            
            if density > 127:  # 0.5 de 255
                voxel = (x, y, slice_idx)
                voxels.append(voxel)

voxels = np.array(voxels)


point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(voxels)


o3d.io.write_point_cloud("/mnt/d/Projetos/Nerf/Instant-NGP-for-RTX-3000-and-4000/output_single.pcd", point_cloud)




# %%


####VECTORR!


from PIL import Image
import numpy as np
import open3d as o3d
Image.MAX_IMAGE_PIXELS = None

image_path = '/mnt/d/data/maq_0407_sam/single/images.density_slices_608x416x608.png'

image = Image.open(image_path)
image_array = np.array(image)

slice_largura = 608
slice_altura = 416
slices = 608

#dim da img
im_largura, im_altura = image.size

#slices por linha e coluna
slices_linha = im_largura // slice_largura
slices_coluna = im_altura // slice_altura

#inicializar uma lista para armazenar os voxels
voxels = []

#calcular offsets para todas as slices
x_offsets = np.tile(np.arange(slices_linha) * slice_largura, slices_coluna)
y_offsets = np.repeat(np.arange(slices_coluna) * slice_altura, slices_linha)

#transformar a imagem em uma máscara booleana com base na densidade
mask = image_array > 127  #0.5 de 255

for slice_idx in range(slices):
    x_offset = x_offsets[slice_idx]
    y_offset = y_offsets[slice_idx]

    #slice atual usando  numpy
    slice_mask = mask[y_offset:y_offset + slice_altura, x_offset:x_offset + slice_largura]

    #coordenadas dos pontos com densidade > 127
    y_coords, x_coords = np.where(slice_mask)

    #adicionar os voxels correspondentes a esta slice
    slice_voxels = np.stack((x_coords, y_coords, np.full_like(x_coords, slice_idx)), axis=-1)
    voxels.append(slice_voxels)

#concatenar todos os voxels
voxels = np.concatenate(voxels, axis=0)


point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(voxels)


o3d.io.write_point_cloud("/mnt/d/Projetos/Nerf/Instant-NGP-for-RTX-3000-and-4000/output_single.pcd", point_cloud)






# %%
