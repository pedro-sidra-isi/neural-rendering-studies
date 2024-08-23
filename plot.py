#%%
import numpy as np
import matplotlib.pyplot as plt


data_gaussian = {
    "Avg dist.": 0.0055,
    "Sigma": 0.042,
    "Max error": 0.071
}

data_nerf = {
    "Avg dist.": 0.026,
    "Sigma": 0.09,
    "Max error": 0.068
}

data_3DGS_1903= {
    "Avg dist.": 0.0107,
    "Sigma": 0.04,
    "Max error": 0.068
}


labels = np.array(list(data_gaussian.keys()))


values_gaussian = np.array(list(data_gaussian.values()))
values_nerf = np.array(list(data_nerf.values()))
values_3DGS_1903 = np.array(list(data_3DGS_1903.values()))

x = np.arange(len(labels))
width = 0.3  
fig, ax = plt.subplots()


rects1 = ax.bar(x - width, values_gaussian, width, label='Gaussian Splatting')
rects2 = ax.bar(x, values_nerf, width, label='NeRF Mesh')
rects3 = ax.bar(x + width, values_3DGS_1903, width, label='3DGS 21/03')


ax.set_ylabel('Values')
ax.set_title('Comparison Sets')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.xticks(rotation=45, ha='right')
plt.show()

# %%
