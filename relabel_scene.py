import argparse
import scipy.spatial as sp
import numpy as np
import pandas as pd
import open3d as o3d
from pypcd import pypcd
from art_utils.relabel import interpolate_columns_nn


# get args in case jsut downsample_all is called
def getArgs():
    parser = argparse.ArgumentParser(
        description="Take a labeled scene and a matching, unlabeled scene and apply the closest "
        "labels to the unlabeled scene."
    )
    parser.add_argument(
        "unlabeled_scene", help="Path to the matching scene that does not have labels"
    )
    parser.add_argument("labeled_scene", help="Path to the scene that has labels")
    return parser.parse_args()


def o3d_pcloud(xyz):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    return pc


def compute_fpfh(pcd_down, voxel_size=0.02):
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_fpfh


def register_ICP(source_xyz, target_xyz, threshold=0.01):
    source = o3d_pcloud(source_xyz)
    target = o3d_pcloud(target_xyz)

    source_down = source.uniform_down_sample(
        every_k_points=100
    )  # .voxel_down_sample(voxel_size=0.02)
    target_down = target.uniform_down_sample(
        every_k_points=100
    )  # .voxel_down_sample(voxel_size=0.02)

    print("Global registration...")
    global_register = (
        o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down,
            target_down,
            compute_fpfh(source_down),
            compute_fpfh(target_down),
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=threshold
            ),
        )
    )

    source_down = source_down.transform(global_register.transformation)

    print("Refine (local) registration...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_correspondence_distance=threshold,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    aligned_source = source.transform(global_register.transformation).transform(
        reg_p2p.transformation
    )
    return np.asarray(aligned_source.points)


def load_pcd(path) -> pd.DataFrame:
    return pd.DataFrame(pypcd.PointCloud.from_path(path).pc_data)


if __name__ == "__main__":
    args = getArgs()

    print(f"loading {args.unlabeled_scene}")
    unlabeled = load_pcd(args.unlabeled_scene)

    print(f"loading {args.labeled_scene}")
    labeled = load_pcd(args.labeled_scene)

    print("relabeling...")
    relabeled_scene = interpolate_columns_nn(
        unlabeled, source=labeled, columns=["instance", "label", "old_label"]
    )
    pypcd.pandas_to_pypcd(relabeled_scene).save_pcd("relabel/test.pcd")
