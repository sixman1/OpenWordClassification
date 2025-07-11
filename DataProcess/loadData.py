from DataProcess.genLoader import GenDataLoader, farthest_point_sampling
from DataProcess.modelnetLoader import ModelNetDataLoader
from DataProcess.mcLoader import McDataLoader
from DataProcess.genmcLoader import GenMcDataLoader
from DataProcess.genLoader_prompt import GenData_Prompts_Loader
from DataProcess.ScanObjectNNLoader import ScanObjectNNDataLoader
from DataProcess.GenScanObjectNNLoader import GenScanObjectNNDataLoader

import torch


def data_load(
        DATA_PATH='./Dataset/modelnet40_normal_resampled',
        DATA_PATH_GEN='./Dataset/',   #gen10 prompts_gen10
        DATA_PATH_GD_GEN='./Dataset/gen10_GD',
        DATA_PATH_MC='./Dataset/mcgill_op_npy',
        DATA_PATH_MC_GEN='./Dataset/mc',
        DATA_PATH_MC_GD_GEN='./Dataset/mc_GD',
        gen_number=6,
        rotation=1,
        type=10,
        prompt=False,
        prompt_id=10,
        mode=0,
        dataset=0
):
    dataset_list = ['gen10', 'prompts_gen10']
    DATA_PATH_GEN = f'{DATA_PATH_GEN}{dataset_list[dataset]}'
    if prompt:
        gen_dataset = GenData_Prompts_Loader(root=DATA_PATH_GEN, npoint=1024, split="test", gen_number=gen_number,
                                type=type, rotation=rotation, prompt_id=prompt_id)
        gen_dataloader = torch.utils.data.DataLoader(dataset=gen_dataset, batch_size=32, shuffle=True, num_workers=0)
    else:
        gen_dataset = GenDataLoader(root=DATA_PATH_GEN, npoint=1024, split="test", gen_number=gen_number,
                                    type=type, rotation=0)
        gen_dataloader = torch.utils.data.DataLoader(dataset=gen_dataset, batch_size=32, shuffle=True, num_workers=0)

    test_dataset = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split="test",
                                      type=type, rotation=rotation, mode=mode)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=0)

    mc_dataset = McDataLoader(root=DATA_PATH_MC, npoint=1024, type="xyz", rotation=rotation)  # xyz or faces
    mc_dataloader = torch.utils.data.DataLoader(dataset=mc_dataset, batch_size=32, shuffle=True, num_workers=0)

    gen_mc_dataset = GenMcDataLoader(root=DATA_PATH_MC_GEN, npoint=1024, gen_number=gen_number,
                                     type="xyz", rotation=rotation)  # xyz or faces
    gen_mc_dataloader = torch.utils.data.DataLoader(dataset=gen_mc_dataset, batch_size=32, shuffle=True, num_workers=0)

    GD_gen_dataset = GenDataLoader(root=DATA_PATH_GD_GEN, npoint=1024, split="test", gen_number=gen_number,
                                   type=10, rotation=0)
    GD_gen_dataloader = torch.utils.data.DataLoader(dataset=GD_gen_dataset, batch_size=32, shuffle=True, num_workers=0)

    GD_gen_mc_dataset = GenMcDataLoader(root=DATA_PATH_MC_GD_GEN, npoint=1024, gen_number=gen_number,
                                        type="xyz", rotation=rotation)  # xyz or faces
    GD_gen_mc_dataloader = torch.utils.data.DataLoader(dataset=GD_gen_mc_dataset, batch_size=32, shuffle=True,
                                                       num_workers=0)

    return gen_dataloader, test_dataloader, mc_dataloader, gen_mc_dataloader, GD_gen_dataloader, GD_gen_mc_dataloader


def load_40data(
        DATA_PATH='./Dataset/modelnet40_normal_resampled',
        DATA_PATH_GEN='./Dataset/gen40',
        gen_number=5,
        rotation=1,
        prompt=False,
        prompt_id=0,
        object_name="bench"
):
    if prompt:
        gen_dataset = GenData_Prompts_Loader(root=DATA_PATH_GEN, npoint=1024, split="test", gen_number=gen_number,
                                type=40, rotation=rotation, prompt_id=prompt_id, object_name=object_name)
        gen_dataloader = torch.utils.data.DataLoader(dataset=gen_dataset, batch_size=32, shuffle=True, num_workers=0)
    else:
        gen_dataset = GenDataLoader(root=DATA_PATH_GEN, npoint=1024, split="test", gen_number=gen_number,
                                    type=40, rotation=1)
        gen_dataloader = torch.utils.data.DataLoader(dataset=gen_dataset, batch_size=32, shuffle=True, num_workers=0)

    test_dataset = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split="test",
                                      type=40, rotation=rotation)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=0)
    return test_dataloader, gen_dataloader

def load_scanobjectnn(
        DATA_PATH='./Dataset/scanObjectNN',
        DATA_PATH_GEN='./Dataset/scanObjectNN',
        gen_number=7,
        rotation=1,
        prompt_id=0,
        object_name="bench",
        mode=0
):
    test_dataset = ScanObjectNNDataLoader(root=DATA_PATH, npoint=1024, split="test", mode=mode, type=15)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=True, num_workers=0)
    gen_dataset = GenScanObjectNNDataLoader(root=DATA_PATH_GEN, npoint=1024, split="test", gen_number=gen_number,
                                         type=15, rotation=rotation)
    gen_dataloader = torch.utils.data.DataLoader(dataset=gen_dataset, batch_size=32, shuffle=True, num_workers=0)

    return test_dataloader, gen_dataloader