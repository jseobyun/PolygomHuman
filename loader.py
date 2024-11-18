import os
import trimesh
import numpy as np
from utils import parse_xml, apply_T


class PolygomHumanLoader():
    def __init__(self, root):
        split_names = sorted(os.listdir(root))
        split_dirs = [os.path.join(root, split_name) for split_name in split_names]


        self.subj_dirs = []
        for split_dir in split_dirs:
            subj_names = sorted(os.listdir(split_dir))
            for subj_name in subj_names:
                self.subj_dirs.append(os.path.join(split_dir, subj_name))

    def load_mesh(self, obj_path):
        obj = trimesh.load(obj_path)
        return obj

    def load_cameras(self, cam_path, coord_path):
        xml_dict = parse_xml(cam_path)
        coord_changer = np.load(coord_path)

        scaler = coord_changer[-1, -1]
        coord_changer = coord_changer / scaler

        poses_normalized = []
        cam_names = []
        Ks = []
        dists = []
        for img_name in list(xml_dict.keys()):
            if len(xml_dict[img_name].keys()) != 3:
                continue
            pose = xml_dict[img_name]["pose"]
            K = xml_dict[img_name]["K"]
            dist = xml_dict[img_name]["dists"]
            cam_names.append(img_name)
            Ks.append(K)
            dists.append(dist)
            pose_normalized = np.matmul(coord_changer, pose)
            pose_normalized[:3, -1] *= scaler
            poses_normalized.append(pose_normalized) # SE3 4x4, cam-to-world T_world_cam

        return cam_names, Ks, dists, poses_normalized


    def __len__(self):
        return len(self.subj_dirs)

    def __getitem__(self, index):
        subj_dir = self.subj_dirs[index]
        obj_path = os.path.join(subj_dir, "mesh.obj")
        cam_path = os.path.join(subj_dir, "cameras.xml")
        coord_path = os.path.join(subj_dir, "coord_changer.npy")
        mesh = self.load_mesh(obj_path)
        cam_names, Ks, dists, poses = self.load_cameras(cam_path, coord_path)


        return mesh, cam_names, Ks, dists, poses




