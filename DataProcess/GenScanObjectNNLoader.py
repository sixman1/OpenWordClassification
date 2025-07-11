import os
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

from scipy.spatial import distance_matrix


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


class GenScanObjectNNDataLoader(Dataset):
    def __init__(self,
                 root,
                 npoint=1024,
                 split="train",
                 gravity_dim=2,
                 type=10,
                 gen_number=3,
                 rotation=1
                 ):
        self.root = root  # 数据集的根目录
        self.npoint = npoint  # 每个点云数据的采样点个数
        self.gravity_dim = gravity_dim
        # 记录40个类别名称的文件
        self.catfile = os.path.join(self.root, f"scanobjectnn_shape_names.txt")
        self.cat = [line.strip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # print(f'self.classes:{self.classes}')
        print('-----ScanObjectNN Dataset!-----')
        self.nviews = 1
        self.r = rotation

        shape_ids = {}
        shape_ids["test"] = [line.strip() for line in open(os.path.join(self.root, f"scanobjectnn_shape_names.txt"))]

        filename = [line.strip() for line in open(os.path.join(self.root, f"mongen{gen_number}.txt"))]
        # print(f'filename:{filename}')
        shape_names = ["_".join(x.split("_")[0:-1]) for x in filename]
        # print(f'shape_names:{shape_names}')

        self.datapath = [(shape_names[i], os.path.join(self.root, filename[i]) + ".ply") for i in range(len(filename))]

        # print("The size of %s data is %d" % (split, len(self.datapath)))

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        cls = self.classes[self.datapath[index][0]]
        cls = np.array([cls]).astype(np.int32)

        ply_file_path = self.datapath[index][1]
        point_cloud = o3d.io.read_point_cloud(ply_file_path)
        points = np.asarray(point_cloud.points).astype(np.float32)

        point_set = farthest_point_sampling(points, 1024)
        point_set = xyz_normalize(point_set)
        if self.r == 1:
            point_set = apply_rotation(point_set)

        data = {
            'pos': point_set.T,
            'cls': cls
        }

        return data

# if __name__ == "__main__":
#     DATA_PATH = 'E:\\Project\\Pre-training\\TAP-main\\examples\\classification\\gpt'
#     test_dataset = GenMcDataLoader(root=DATA_PATH, npoint=1024, split="test", gen_number=0)
#     test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=True, num_workers=0)
#     for batch_id, data in enumerate(test_dataloader):
#         print(data.keys())
#         print(data['pos'].shape)  # 8*1024*3
#         print(data['cls'].shape)  # 8*1
#         break
