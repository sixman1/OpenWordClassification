import numpy as np
import torch

from tetrasphere.models.tetrasphere import TetraSphere_cls
from ShowResults import cal_acc
from DataProcess.loadData import data_load, load_40data, load_scanobjectnn


def make_object(**kwargs):
    cls = kwargs['cls']
    kwargs = {k: v for k, v in kwargs.items() if k != 'cls'}
    obj = cls(**kwargs)
    return obj


def model_load(model_kwargs={}, seed=3, num_epochs=250, batch_size=32, dry_run=False, num_classes=40, ckpt_path=''):
    model_args = dict(cls=TetraSphere_cls,
                      k=20,
                      init_mode=None,
                      output_channels=num_classes,
                      fix_tetrasphere=model_kwargs.get("fix_tetrasphere", False),
                      normalized_spheres=model_kwargs.get("normalized_spheres", False),
                      num_spheres=model_kwargs["num_spheres"],
                      sphere_pooling=model_kwargs.get("sphere_pooling", "equi_max_norm"))

    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['state_dict']  # 获取模型权重

    # 通过 make_object 创建模型
    model = make_object(**model_args)  # 使用 model_args 中的配置字典创建模型
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('model.', '')  # 去掉 'model.' 前缀
        new_state_dict[new_key] = v
    # 加载模型的权重
    model.load_state_dict(new_state_dict, strict=False)  # 将训练时的权重应用到模型中

    print('486')

    return model

@torch.no_grad()
def exFeature(model, data_loader, device):
    model.eval()
    features = []
    labels = []

    for idx, data in enumerate(data_loader):
        feature = model(data['pos'].to(device).float())
        features.append(feature)
        labels.append(data['cls'])

    return features, labels

def get_scanobjectnn_feature(model, gen_dataloader, test_dataloader, device, gen_number=7, rotation=1,
                             type=10, save_file=True, error_flag=False):
    gen_features, gen_labels = exFeature(model, gen_dataloader, device)
    gen_features = torch.cat(gen_features, dim=0)
    gen_labels = torch.cat(gen_labels, dim=0)

    features, labels = exFeature(model, test_dataloader, device)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    print(f'--- ScanObjectNN Results ---')
    show_r2(features=features, gen_features=gen_features, labels=labels, gen_labels=gen_labels, cls_num=15)

    print('ScanObjectNN Done!')


def get_modelnet_feature(model, gen_dataloader, GD_gen_dataloader, test_dataloader, device, gen_number=5, rotation=1,
                         type=10, save_file=True, error_flag=False):
    if rotation == 1:
        r_text = ''
    else:
        r_text = '_nr'
    gen_features, gen_labels = exFeature(model, gen_dataloader, device)
    gen_features = torch.cat(gen_features, dim=0)
    gen_labels = torch.cat(gen_labels, dim=0)

    modelnet_features, modelnet_labels = exFeature(model, test_dataloader, device)
    modelnet_features = torch.cat(modelnet_features, dim=0)
    modelnet_labels = torch.cat(modelnet_labels, dim=0)

    if type == 10:
        GD_gen_features, GD_gen_labels = exFeature(model, GD_gen_dataloader, device)
        GD_gen_features = torch.cat(GD_gen_features, dim=0)
        GD_gen_labels = torch.cat(GD_gen_labels, dim=0)
        print(f'--- Modelnet{type} Results ---')
        show_r(modelnet_features, gen_features, GD_gen_features, modelnet_labels, gen_labels, GD_gen_labels,
               cls_num=type, top_n=1, gen_number=gen_number, error_flag=error_flag)

    else:
        print(f'--- Modelnet{type} Results ---')
        show_r2(modelnet_features, gen_features, modelnet_labels, gen_labels, cls_num=type, top_n=1)


    print(f'Modelnet{type} Done!')


def get_mcgill_feature(model, mc_dataloader, gen_mc_dataloader, GD_gen_mc_dataloader, device, gen_number=5, rotation=1):
    if rotation == 1:
        r_text = ''
    else:
        r_text = '_nr'
    mc_features, mc_labels = exFeature(model, mc_dataloader, device)
    mc_features = torch.cat(mc_features, dim=0)
    mc_labels = torch.cat(mc_labels, dim=0)

    gen_mc_features, gen_mc_labels = exFeature(model, gen_mc_dataloader, device)
    gen_mc_features = torch.cat(gen_mc_features, dim=0)
    gen_mc_labels = torch.cat(gen_mc_labels, dim=0)

    GD_gen_mc_features, GD_gen_mc_labels = exFeature(model, GD_gen_mc_dataloader, device)
    GD_gen_mc_features = torch.cat(GD_gen_mc_features, dim=0)
    GD_gen_mc_labels = torch.cat(GD_gen_mc_labels, dim=0)

    print('--- McGill Results ---')

    show_r(mc_features, gen_mc_features, GD_gen_mc_features, mc_labels, gen_mc_labels, GD_gen_mc_labels,
           cls_num=14, top_n=1, gen_number=gen_number)

    print('McGill Done!')


def get_mn40_feature(model, gen_dataloader, test_dataloader, device, gen_number=5, rotation=1,
                         type=40, save_file=True, error_flag=False, prompt_id=5):
    gen_features, gen_labels = exFeature(model, gen_dataloader, device)
    gen_features = torch.cat(gen_features, dim=0)
    gen_labels = torch.cat(gen_labels, dim=0)

    modelnet_features, modelnet_labels = exFeature(model, test_dataloader, device)
    modelnet_features = torch.cat(modelnet_features, dim=0)
    modelnet_labels = torch.cat(modelnet_labels, dim=0)

    print(f"gen_number: {gen_number}, prompt_id: {prompt_id}\n")
    show_r2(modelnet_features, gen_features, modelnet_labels, gen_labels, cls_num=40, top_n=1, gen_number=gen_number)
    print(f'Modelnet{type} Done!\n')

def show_r(features, gen_features, GD_gen_features, labels, gen_labels, GD_gen_labels, cls_num, top_n=1, gen_number=5,
           error_flag=False):
    m1_feature = features.detach().cpu()
    m2_feature = gen_features.detach().cpu()
    m3_feature = GD_gen_features.detach().cpu()
    m1_label = labels.detach().cpu().numpy()
    m2_label = gen_labels.detach().cpu().numpy()
    m3_label = GD_gen_labels.detach().cpu().numpy()

    m1_feature = np.asarray(m1_feature).astype(np.float32)
    m2_feature = np.asarray(m2_feature).astype(np.float32)
    m3_feature = np.asarray(m3_feature).astype(np.float32)

    mix_feature = np.concatenate((m2_feature, m3_feature), axis=0)
    mix_label = np.concatenate((m2_label, m3_label), axis=0)

    print('feature & gen_feature')
    cal_acc(m1_feature, m2_feature, m1_label, m2_label, cls_num=cls_num, top_n=top_n, gen_number=gen_number,
            error_flag=error_flag)
    print('feature & GD_gen_feature')
    cal_acc(m1_feature, m3_feature, m1_label, m3_label, cls_num=cls_num, top_n=top_n, gen_number=gen_number,
            error_flag=error_flag)
    print('feature & mix_feature')
    cal_acc(m1_feature, mix_feature, m1_label, mix_label, cls_num=cls_num, top_n=top_n, gen_number=gen_number,
            error_flag=error_flag)


def show_r2(features, gen_features, labels, gen_labels, cls_num, top_n=1, gen_number=6):
    m1_feature = features.detach().cpu()
    m2_feature = gen_features.detach().cpu()
    m1_label = labels.detach().cpu().numpy()
    m2_label = gen_labels.detach().cpu().numpy()
    m1_feature = np.asarray(m1_feature).astype(np.float32)
    m2_feature = np.asarray(m2_feature).astype(np.float32)

    print('feature & gen_feature')
    cal_acc(m1_feature, m2_feature, m1_label, m2_label, cls_num=cls_num, top_n=top_n, gen_number=gen_number)


def load_model_mn(ckpt_path, device, checkpointID=1):
    if checkpointID == 1:
        ckpt = f'{ckpt_path}cvpr2024_tetrasphere_K=2_zaug_sobjnn_objbg_87_3.ckpt'
        model_mn = model_load(
            model_kwargs=dict(num_spheres=2),
            ckpt_path=ckpt,
            num_epochs=1000,
            num_classes=15
        ).to(device)
    elif checkpointID == 2:
        print('modelnet30_Pretrain')
        ckpt_path = f'{ckpt_path}epoch=249.ckpt'
        model_mn = model_load(model_kwargs=dict(num_spheres=8),
                          ckpt_path=ckpt_path
                          ).to(device)
    else:
        print('modelnet40_Pretrain')
        ckpt_path = f'{ckpt_path}cvpr2024_tetrasphere_K=8_zaug_mn40_90_5.ckpt'
        model_mn = model_load(model_kwargs=dict(num_spheres=8),
                              ckpt_path=ckpt_path
                              ).to(device)
    return model_mn


def load_model_mc(ckpt_path, device, checkpointID=1):
    if checkpointID == 1:
        print('modelnet40_Pretrain')
        model_mc = model_load(model_kwargs=dict(num_spheres=8),
                              ckpt_path=f'{ckpt_path}cvpr2024_tetrasphere_K=8_zaug_mn40_90_5.ckpt'
                              ).to(device)
    elif checkpointID == 2:
        print('modelnet30_Pretrain')
        model_mc = model_load(model_kwargs=dict(num_spheres=8),
                              ckpt_path=f'{ckpt_path}epoch=249.ckpt'
                              ).to(device)
    return model_mc


def main_mn(ckpt_path, gen_number=7, rotation=1, type=10, prompt_id=5, prompt=False, save_file=False, checkpointID=1, mode=0, dataset=0):

    gen_dataloader, test_dataloader, mc_dataloader, gen_mc_dataloader, GD_gen_dataloader, \
    GD_gen_mc_dataloader = data_load(
        gen_number=gen_number, rotation=rotation, type=type, prompt_id=prompt_id, prompt=prompt, mode=mode, dataset=dataset
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model_mn = load_model_mn(ckpt_path, device=device, checkpointID=checkpointID)

    get_modelnet_feature(model_mn, gen_dataloader, GD_gen_dataloader, test_dataloader, device,
                         gen_number=gen_number, rotation=rotation, type=type, save_file=save_file, error_flag=False)

def main_scanobjectnn(mode=0):
    ckpt_path = '/gpfs/work/int/xiyang/Xinzhe_xia/Project/tetrasphere-main/tetrasphere/weights/'
    test_dataloader, gen_dataloader = load_scanobjectnn(mode=mode)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_mn(ckpt_path, device=device, checkpointID=3)
    get_scanobjectnn_feature(model, gen_dataloader, test_dataloader, device)

def main_mn40(gen_number=6, rotation=1, prompt=False, prompt_id=0, object_name="bench"):
    ckpt_path = f'./tetrasphere/weights/'
    test_dataloader, gen_dataloader = load_40data(
        gen_number=gen_number,
        rotation=rotation,
        prompt=prompt,
        prompt_id=prompt_id,
        object_name=object_name
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_mn = load_model_mn(ckpt_path, device=device)
    get_mn40_feature(model_mn, gen_dataloader, test_dataloader, device, gen_number, rotation)


def main_mc(args, gen_number=6, rotation=1, type=10, save_file=False, checkpointID=1):

    if type == 10:
        gen_dataloader, test_dataloader, mc_dataloader, gen_mc_dataloader, GD_gen_dataloader, \
        GD_gen_mc_dataloader = data_load(gen_number=gen_number, rotation=rotation, type=type)
    else:
        gen_dataloader, test_dataloader = load_40data(gen_number=gen_number, rotation=1)
        mc_dataloader = []
        gen_mc_dataloader = []
        GD_gen_dataloader = []
        GD_gen_mc_dataloader = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_mc = load_model_mc(ckpt_path, device=device, checkpointID=checkpointID)

    get_mcgill_feature(model_mc, mc_dataloader, gen_mc_dataloader, GD_gen_mc_dataloader, device, gen_number=gen_number)


if __name__ == "__main__":

    mode_list = ['None', 'None', 'Noise', 'Mirror', 'Stretching', 'Remove', 'Jittering']
    ckpt_path = 'path/to/your/pretrain_model'


    main_mn(ckpt_path, gen_number=7, prompt=False, prompt_id=5, save_file=False, checkpointID=2, mode=2, dataset=1)

    print('486')
