
import os
import torch, torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils import data
import sys
import yaml
sys.path.append('../..')
import shap
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from gradientshap import Gradient
from datasets import get_dataset
from datasets import NAMES as DATASET_NAMES
from argparse import ArgumentParser

def test_der():
    sys.argv = ['test', '--model','der','--dataset','seq-cifar10']

def get_shap_args():
    parser = ArgumentParser(description='Calculate SHAP value', allow_abbrev=False)
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--dataset', type=str, help='used datasets')
    parser.add_argument('--device', type=str, default='cuda'  if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--config', type=str, help='path to your config file')
    
    args = parser.parse_args()
    
    return args

def main(args):
    print(f'{args.model}')
    dataset = get_dataset(args)
    model = dataset.get_backbone()
    # if args.model == 'foster':
    #     model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    for class_index in range(model.num_classes):
        test_loader = dataset.get_shapley_data()
        
        local_progress=tqdm(test_loader, desc=f'Class {class_index}/{model.num_classes}', disable=False)
        for idx, (inputs, targets) in enumerate(local_progress):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            for model_num in range(int(class_index / dataset.N_CLASSES_PER_TASK), dataset.N_TASKS):
                model_path = f'data/results/{args.dataset}/{args.model}/TASK_{model_num}.pth'
                checkpoint = torch.load(model_path)
                model.load_state_dict({k[4:]:v for k,v in checkpoint.items() if "net" in k}, strict=False)
                model.to(args.device)
                model.eval()
                
                selected_classes = [class_index]
                e = Gradient(model, inputs, local_smoothing=0) 
                shap_values, index = e.shap_values(inputs, nsamples=100, ranked_outputs=selected_classes, output_rank_order="custom")#用所有的样本计算shapley value
                
                shap_values_array = np.array(shap_values).squeeze()
                
                shap_save_path = f'shapley_value_conv/{args.dataset}/{args.model}/model{model_num}'
                if not os.path.exists(shap_save_path):
                    os.makedirs(shap_save_path)
                np.save(f'{shap_save_path}/class{class_index}_shap_values.npy',shap_values_array)

            ...
    ...

if __name__ == '__main__':
    # test_der()
    args = get_shap_args()
    main(args)

