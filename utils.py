import cv2
import time
import datetime
import numpy as np
import xml.etree.ElementTree as ET

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    intrinsics = root[0][0]
    extrinsics = root[0][2]

    xml_dict = {}
    for intrinsic in intrinsics:
        try:
            img_name = intrinsic.attrib["label"]
            intr_values = intrinsic[2]
            img_w = float(intr_values[0].attrib["width"])
            img_h = float(intr_values[0].attrib["height"])

            f = float(intr_values[1].text)
            cx, cy = float(intr_values[2].text), float(intr_values[3].text)
            k1, k2, k3 = float(intr_values[4].text), float(intr_values[5].text), float(intr_values[6].text)
            p1, p2 = float(intr_values[7].text), float(intr_values[8].text)
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = f
            K[1, 1] = f
            K[0, 2] = img_w / 2 + cx
            K[1, 2] = img_h / 2 + cy
            dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
            xml_dict[img_name] = {}
            xml_dict[img_name]["K"] = K
            xml_dict[img_name]["dists"] = dist
        except:
            continue

    img_names = list(xml_dict.keys())
    for extrinsic in extrinsics:
        try:
            img_name = extrinsic.attrib["label"]
            if img_name not in img_names:
                continue

            pose = np.asarray(extrinsic[0].text.split(' '), dtype=np.float32)
            pose = pose.reshape(4, 4)
            xml_dict[img_name]["pose"] = pose  # T_gk
        except:
            continue
    return xml_dict

def apply_T(T, points):
    if len(T.shape) != 2 or len(points.shape) != 2:
        raise Exception("ERROR : the dimensions of transformation matrix and points are wrong.")

    points_ = np.matmul(T[:3, :3], points.transpose(1, 0)).transpose(1, 0) + T[:3, -1].reshape(-1, 3)
    return points_
