import math
import os
import torch
import numpy as np
from torch.utils.data import Dataset

from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA


def random_rotation_matrix():
    # 随机生成三个角度
    theta_x = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)

    # 计算绕X轴的旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])

    # 计算绕Y轴的旋转矩阵
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])

    # 计算绕Z轴的旋转矩阵
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])

    # 组合旋转矩阵
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def rotate_point_cloud(point_cloud, index):
    angle = np.load("./random_angle.npy")
    cur_angle = angle[index, :]


    angle_x = np.deg2rad(cur_angle[0])
    angle_y = np.deg2rad(cur_angle[1])
    angle_z = np.deg2rad(cur_angle[2])


    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(angle_x), -np.sin(angle_x)],
                                  [0, np.sin(angle_x), np.cos(angle_x)]])
    rotation_matrix_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                                  [0, 1, 0],
                                  [-np.sin(angle_y), 0, np.cos(angle_y)]])
    rotation_matrix_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                                  [np.sin(angle_z), np.cos(angle_z), 0],
                                  [0, 0, 1]])


    point_cloud_T = point_cloud.T


    point_cloud_T = rotation_matrix_x.dot(point_cloud_T)
    point_cloud_T = rotation_matrix_y.dot(point_cloud_T)
    point_cloud_T = rotation_matrix_z.dot(point_cloud_T)


    rotated_point_cloud = point_cloud_T.T

    return rotated_point_cloud

def apply_rotation(point_cloud):
    rotation_matrix = random_rotation_matrix()
    return np.dot(point_cloud, rotation_matrix.T)


def farthest_point_sampling(point_cloud, num_samples):
    num_points = point_cloud.shape[0]
    distances = np.full(num_points, np.inf)
    selected_points = []

    # 随机选择一个初始点
    initial_index = np.random.randint(0, num_points)
    selected_points.append(initial_index)
    current_distances = distance_matrix(point_cloud, point_cloud[initial_index:initial_index + 1, :])
    distances = np.minimum(distances, current_distances[:, 0])

    while len(selected_points) < num_samples:
        # 选择最远的点
        next_index = np.argmax(distances)
        selected_points.append(next_index)
        current_distances = distance_matrix(point_cloud, point_cloud[next_index:next_index + 1, :])
        distances = np.minimum(distances, current_distances[:, 0])

    return point_cloud[selected_points]


def farthest_point_sampling_v1(point_cloud, num_samples):
    N = point_cloud.shape[0]
    if num_samples >= N:
        return np.arange(N)

    distances = np.full(N, np.inf)
    sampled_indices = np.zeros(num_samples, dtype=int)

    sampled_indices[0] = np.random.randint(N)
    max_distance = 0

    for i in range(1, num_samples):
        distances = np.minimum(distances, np.linalg.norm(point_cloud - point_cloud[sampled_indices[i - 1]], axis=1))
        sampled_indices[i] = np.argmax(distances)

    return point_cloud[sampled_indices]


def xyz_normalize(point):
    """
    point:[[x1,y1,z1],[x2,y2,z2],...,[xn,yn,zn]]
    计算采样点与中心点之间的距离,并且将距离归一化
    """
    centroid = np.mean(point, axis=0)
    point = point - centroid
    dist = np.max(np.sqrt(np.sum(point ** 2, axis=1)))
    point = point / dist
    return point


def add_noise(point_set):
    # 添加高斯噪声
    mean = 0
    std_dev = 0.005  # 噪声标准差，可根据需求调整
    noise = np.random.normal(mean, std_dev, point_set.shape)

    noisy_point_cloud = point_set + noise
    return noisy_point_cloud

def get_mirror(point_set, axis=0):
    rand_axis = np.random.choice([0, 1, 2])
    mirrored = point_set.copy()
    mirrored[:, rand_axis] = -mirrored[:, rand_axis]
    return mirrored

def point_stretching(point_set, x=1.0, y=1.0, z=1.0):
    # 创建缩放矩阵（对 X/Y/Z 分别缩放）
    rand_x = np.random.uniform(0.5, 2.0)
    rand_y = np.random.uniform(0.5, 2.0)
    rand_z = np.random.uniform(0.5, 2.0)
    scaling_matrix = np.diag([rand_x, rand_y, rand_z])  # 对角矩阵
    # 线性变换
    stretched = point_set @ scaling_matrix.T
    return stretched

def remove_random_sphere(points, radius_ratio=0.2):
    """
    在点云中随机移除一个球形区域的点
    :param points: (N, 3) 的点云数组
    :param radius_ratio: 移除区域半径占点云最大范围的比例
    :return: 挖洞后的点云
    """
    # 计算点云的范围和中心
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    center_bounds = (min_bounds + max_bounds) / 2
    max_range = np.linalg.norm(max_bounds - min_bounds)

    # 随机选择球心
    sphere_center = center_bounds + (np.random.rand(3) - 0.5) * max_range * 0.5
    radius = radius_ratio * max_range

    # 计算每个点到球心的距离
    distances = np.linalg.norm(points - sphere_center, axis=1)

    # 保留不在球体中的点
    mask = distances > radius
    return points[mask]

def jitter_point_cloud(points, sigma=0.01, clip=0.05):
    """
    对点云添加随机抖动
    :param points: (N, 3) 的点云数组
    :param sigma: 高斯噪声的标准差
    :param clip: 限制抖动的最大值（防止偏移太大）
    :return: 加了抖动的点云
    """
    jitter = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    return points + jitter

class ModelNetDataLoader(Dataset):
    def __init__(self,
                 root,
                 npoint=1024,
                 split="train",
                 gravity_dim=2,
                 type=10,
                 rotation=1,
                 mode=1
                 ):
        self.root = root  # 数据集的根目录
        self.npoint = npoint  # 每个点云数据的采样点个数
        self.gravity_dim = gravity_dim
        # 记录40个类别名称的文件
        self.catfile = os.path.join(self.root, f"modelnet{type}_shape_names.txt")
        self.cat = [line.strip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        print('-----Modelnet10 Dataset!-----')
        self.nviews = 1
        self.r = rotation
        self.type = type
        self.mode = mode

        shape_ids = {}
        shape_ids["train"] = [line.strip() for line in open(os.path.join(self.root, f"modelnet{type}_train.txt"))]
        shape_ids["test"] = [line.strip() for line in open(os.path.join(self.root, f"modelnet{type}_test.txt"))]

        assert (split == "train" or split == "test")

        shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids[split]]

        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + ".txt") for i
                         in range(len(shape_ids[split]))]

        # print("The size of %s data is %d" % (split, len(self.datapath)))

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        cls = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)
        if self.mode == 2:
            point_set = add_noise(point_set)
        if self.mode == 3:
            point_set = get_mirror(point_set)
        if self.mode == 4:
            point_set = point_stretching(point_set[:, :3])
        if self.mode == 5:
            point_set = remove_random_sphere(point_set[:, :3])
        if self.mode == 6:
            point_set = jitter_point_cloud(point_set[:, :3])
        point_set = farthest_point_sampling(point_set[:, :3], 1024)
        point_set = xyz_normalize(point_set)
        if self.r == 1:
            if self.type == 10:
                point_set = rotate_point_cloud(point_set, index)
            else:
                point_set = apply_rotation(point_set)

        data = {
            'pos': point_set.T,
            'cls': cls
        }

        return data