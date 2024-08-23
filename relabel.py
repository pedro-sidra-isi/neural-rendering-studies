from art_utils.datasets import Dataset
from tqdm import tqdm
from pypcd import pypcd
import scipy.spatial as sp
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def interpolate_columns_nn(target, source, max_dist=0.04, columns=[]) -> pd.DataFrame:
    """
    Populate `columns` on the `target` point cloud with values coming from the nearest neighbors of
    `target` in the `source` point cloud

    :returns target: the target point cloud populated with values from the


    """
    source_xyz = source[["x", "y", "z"]].to_numpy()
    target_xyz = target[["x", "y", "z"]]

    kdtree = sp.KDTree(source_xyz)

    # Why k=2? don't know, but it works with k=2 and not with k=1
    nearest_label_dists, nearest_label_idxs = kdtree.query(target_xyz, k=2, workers=-1)

    valid_neighbors = nearest_label_dists[:, 0] < max_dist

    nearest_label_idxs = nearest_label_idxs[valid_neighbors, 0]

    relabeled_columns = []
    for col in source.columns:
        if col not in columns:
            continue

        relabeled_columns.append(col)
        target_instances = source[col].to_numpy()[nearest_label_idxs]
        target[col] = np.nan
        target.loc[valid_neighbors, col] = target_instances

    return target, relabeled_columns
