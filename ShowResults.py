import heapq

import torch
import numpy as np
import pandas as pd


def calculate_mse(predictions, categories, labels, lc, cls_num=15):
    num_predictions = predictions.shape[0]
    num_categories = categories.shape[0]

    ok = 0
    p = [0] * cls_num
    error = np.zeros((cls_num, cls_num))
    for i in range(num_predictions):
        min_dis = 100
        l_f = -1
        for j in range(num_categories):
            dis = np.sum((predictions[i] - categories[j]) ** 2) / predictions.shape[1]

            if dis < min_dis:
                l_f = j
                min_dis = dis
        if labels[i] == l_f:
            ok += 1
            p[l_f] += 1

    return ok, p, error


def CosineDistance(x, y):
    x = np.array(x)
    y = np.array(y)
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def fun2(feature_m, feature_g, label_m, label_g, cls_num=15, top_n=1):
    num_feature_m = feature_m.shape[0]
    num_feature_g = feature_g.shape[0]
    ok = 0
    p = np.zeros(cls_num)
    error = np.zeros((cls_num, cls_num))
    for i in range(num_feature_m):
        label = label_m[i]
        dis_l = []
        for j in range(num_feature_g):
            dis = np.sum((feature_m[i] - feature_g[j]) ** 2) / feature_m.shape[1]
            dis_l.append(dis)
        largest_indices = [idx for idx, _ in heapq.nsmallest(top_n, enumerate(dis_l), key=lambda x: x[1])]
        if label in label_g[largest_indices]:
            ok += 1
            p[label] += 1

    return ok, p, error

def fun2_error(feature_m, feature_g, label_m, label_g, cls_num=15, top_n=1):
    num_feature_m = feature_m.shape[0]
    num_feature_g = feature_g.shape[0]
    ok = 0
    p = np.zeros(cls_num)
    error = np.zeros((cls_num, cls_num))
    for i in range(num_feature_m):
        min_dis = 100
        l_flag = 1
        label = label_m[i]
        dis_l = []
        for j in range(num_feature_g):
            dis = np.sum((feature_m[i] - feature_g[j]) ** 2) / feature_m.shape[1]
            if dis < min_dis:
                min_dis = dis
                l_flag = label_g[j]

        if label == l_flag:
            ok += 1
            p[label] += 1
        else:
            error[label][l_flag] += 1
    return ok, p, error


def ave_Cosine(m1_feature, m2_feature, m1_label, m2_label, cls_num=15, top_n=1):
    df = pd.DataFrame(m2_feature)
    df['label'] = m2_label

    average_features = df.groupby('label').mean()
    average_features_np = average_features.values
    p = np.zeros(cls_num)
    ok = 0
    for i in range(len(m1_feature)):
        l_f = m1_label[i]
        dis_l = []
        for j in range(len(average_features_np)):
            dis = CosineDistance(m1_feature[i], average_features_np[j])
            dis_l.append(dis)
        largest_indices = [idx for idx, _ in heapq.nsmallest(top_n, enumerate(dis_l), key=lambda x: x[1])]
        if l_f in m2_label[largest_indices]:
            ok += 1
            p[l_f] += 1

    return ok, p


def mygo_Cosine(m1_feature, m2_feature, m1_label, m2_label, cls_num=15, top_n=1):
    p = np.zeros(cls_num)
    ok = 0
    for i in range(len(m1_feature)):
        l_f = m1_label[i]
        dis_l = []
        for j in range(len(m2_feature)):
            dis = CosineDistance(m1_feature[i], m2_feature[j])
            # dis = CosineDistance(feature_modelnet10t[i], feature_modelnet10t[modelnet10_cl[j]])
            dis_l.append(dis)
        largest_indices = [idx for idx, _ in heapq.nsmallest(top_n, enumerate(dis_l), key=lambda x: x[1])]
        if l_f in m2_label[largest_indices]:
            ok += 1
            p[l_f] += 1

    return ok, p

def get_shape_files_name(num=5, flag1=0, flag2=1, GD1=0, GD2=0):
    file_name = ['modelnet', 'mc', 'tap_modelnet10', 'tap_mc']
    GD1_text = ""
    GD2_text = ""
    if flag1 in [0, 1]:
        feature_text1 = 'feature'
    else:
        feature_text1 = 'features'
    if flag2 in [0, 1]:
        feature_text2 = 'feature'
    else:
        feature_text2 = 'features'
    if GD1 == 1:
        GD1_text = "_GD"
    if GD2 == 1:
        GD2_text = "_GD"
    f1_path = f'gen10/{file_name[flag1]}_{feature_text1}_g{num}.pth'
    l1_path = f'gen10/{file_name[flag1]}_labels_g{num}.pth'
    fg1_path = f'gen10/{file_name[flag1]}{GD1_text}_gen_{feature_text1}_g{num}.pth'
    lg1_path = f'gen10/{file_name[flag1]}{GD1_text}_gen_labels_g{num}.pth'
    f2_path = f'gen10/{file_name[flag2]}_{feature_text2}_g{num}.pth'
    l2_path = f'gen10/{file_name[flag2]}_labels_g{num}.pth'
    fg2_path = f'gen10/{file_name[flag2]}{GD2_text}_gen_{feature_text2}_g{num}.pth'
    lg2_path = f'gen10/{file_name[flag2]}{GD2_text}_gen_labels_g{num}.pth'

    return f1_path, l1_path, fg1_path, lg1_path, f2_path, l2_path, fg2_path, lg2_path


def get_feature(
        flag1=0,
        flag2=1,
        f1_path='gen10/modelnet_feature_g5.pth',
        l1_path='gen10/modelnet_labels_g5.pth',
        fg1_path='gen10/modelnet_gen_feature_g5.pth',
        lg1_path='gen10/modelnet_gen_labels_g5.pth',
        f2_path='gen10/mc_feature_g5.pth',
        l2_path='gen10/mc_labels_g5.pth',
        fg2_path='gen10/mc_gen_feature_g5.pth',
        lg2_path='gen10/mc_gen_labels_g5.pth',
):
    feature1 = torch.load(f1_path).detach().cpu()
    labels1 = torch.load(l1_path).detach().cpu().numpy()
    if flag1 == 2:
        feature1 = torch.max(feature1, dim=1)[0]
    feature1 = np.asarray(feature1).astype(np.float32)

    feature_gen1 = torch.load(fg1_path).detach().cpu()
    labels_gen1 = torch.load(lg1_path).detach().cpu().numpy()
    if flag1 == 2:
        feature_gen1 = torch.max(feature_gen1, dim=1)[0]
    feature_gen1 = np.asarray(feature_gen1).astype(np.float32)
    labels1 = labels1.reshape(-1)

    feature2 = torch.load(f2_path).detach().cpu()
    labels2 = torch.load(l2_path).detach().cpu().numpy()
    if flag2 == 3:
        feature2 = torch.max(feature2, dim=1)[0]
    feature2 = np.asarray(feature2).astype(np.float32)

    feature_gen2 = torch.load(fg2_path).detach().cpu()
    labels_gen2 = torch.load(lg2_path).detach().cpu().numpy()
    if flag2 == 3:
        feature_gen2 = torch.max(feature_gen2, dim=1)[0]
    feature_gen2 = np.asarray(feature_gen2).astype(np.float32)
    labels2 = labels2.reshape(-1)

    return feature1, labels1, feature_gen1, labels_gen1, feature2, labels2, feature_gen2, labels_gen2


def cal_acc(m1_feature, m2_feature, m1_label, m2_label, cls_num=10, top_n=3, gen_number=5, error_flag=False):
    if cls_num == 10:
        name = 'Modelnet10'
        count = [50, 100, 100, 86, 86, 100, 86, 100, 100, 100]
    elif cls_num == 14:
        name = 'McGill'
        count = [10, 7, 10, 7, 4, 8, 7, 8, 7, 11, 9, 9, 11, 7]
    elif cls_num == 40:
        name = 'Modelnet40'
        count = [100, 50, 100, 20, 100, 100, 20, 100, 100, 20,
                 20, 20, 86, 20, 86, 20, 100, 100, 20, 20, 20,
                 100, 100, 86, 20, 100, 100, 20, 100, 20, 100,
                 20, 20, 100, 20, 100, 100, 100, 20, 20]
    elif cls_num == 15:
        name = 'ScanObjectNN'
        count = [17, 40, 28, 75, 78, 30, 42, 42, 49, 54, 22, 21, 24, 42, 17]
    else:
        print(f'ERROR: value cls_num = {cls_num}\t:(')
        print('cls_num should be 10, 14, 15 or 40!!!')
        return 0

    print(f'{name}, TOP {top_n}, Gen {gen_number}: ')

    v = np.vectorize(lambda x: f'{x:.1f}')

    if error_flag:
        ok, p, error = fun2_error(m1_feature, m2_feature, m1_label, m2_label, cls_num=cls_num, top_n=1)
    else:
        ok, p, error = fun2(m1_feature, m2_feature, m1_label, m2_label, cls_num=cls_num, top_n=top_n)
    pp = v(p / count * 100)
    mse_oacc = ok / len(m1_feature)
    mse_macc = np.sum(p / count) / cls_num

    print(f'MSE oACC: {mse_oacc * 100:.1f}')
    print(f'MSE mACC: {mse_macc * 100:.1f}')
    print(f'{pp}')
    if error_flag:
        print(f'c_matrix:\n{error}')

    ok, macc = mygo_Cosine(m1_feature, m2_feature, m1_label, m2_label, cls_num=cls_num, top_n=top_n)
    pp = v(macc / count * 100)
    mse_oacc = ok / len(m1_feature)
    mse_macc = np.sum(macc / count) / cls_num
    print(f'COS oACC: {mse_oacc * 100:.1f}')
    print(f'COS mACC: {mse_macc * 100:.1f}')
    print(f'{pp}')
    # print("BanG Dream! It's MyGO!!!!!")


def main(num=5, flag1=0, flag2=1, top_n=3):

    file_name = ['modelnet', 'mc', 'tap_modelnet10', 'tap_mc']
    f1_path, l1_path, fg1_path, lg1_path, f2_path, l2_path, fg2_path, lg2_path = \
        get_shape_files_name(num=num, flag1=flag1, flag2=flag2, GD1=0, GD2=0)

    feature1, labels1, feature_gen1, labels_gen1, feature2, \
    labels2, feature_gen2, labels_gen2 = get_feature(
        flag1, flag2,
        f1_path=f1_path,
        l1_path=l1_path,
        fg1_path=fg1_path,
        lg1_path=lg1_path,
        f2_path=f2_path,
        l2_path=l2_path,
        fg2_path=fg2_path,
        lg2_path=lg2_path,
    )


    cal_acc(feature2, feature_gen2, labels2, labels_gen2, cls_num=14, top_n=top_n)

    l = {'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'night_stand': 6, 'sofa': 7,
         'table': 8,
         'toilet': 9}
    l_mc = {'ant': 0, 'bird': 1, 'crab': 2, 'dinosaur': 3, 'dolphin': 4, 'fish': 5, 'hand': 6, 'octopus': 7,
            'pliers': 8, 'quadruped': 9, 'snake': 10, 'spectacle': 11, 'spider': 12, 'teddy': 13}

    print('486')


if __name__ == "__main__":
    for i in range(3):
        main(top_n=i+1)
