
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
from datasets.seq_tinyimagenet import TinyImagenet
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_der():
    sys.argv = ['test', '--model','icarl','--dataset','seq-cifar100']
    

def get_shap_args():
    parser = ArgumentParser(description='Calculate SHAP value', allow_abbrev=False)
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--dataset', type=str, help='used datasets')
    parser.add_argument('--device', type=str, default='cuda'  if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--config', type=str, help='path to your config file')
    
    args = parser.parse_args()
    
    return args

def main(args):
    print(f'{args.model} {args.dataset}')
    dataset = get_dataset(args)
    # ##CIFAR10
    if args.dataset == 'seq-cifar10':
        trans = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2470, 0.2435, 0.2615))])
        test_dataset = datasets.CIFAR10(root='/home/lilipan/ling/CIFAR10', train=False, download=False, transform=trans)
        task_num = 5
        task_class = 2
    elif args.dataset == 'seq-cifar100':
        #CIFAR100
        trans = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))])
        test_dataset = datasets.CIFAR100(root='/home/lilipan/ling/CIFAR100', train=False, download=False, transform=trans)
        task_num = 10
        task_class = 10
    else:
        ###Tinyimg
        trans = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4480, 0.3975),
                                    (0.2770, 0.2691, 0.2821))])
        test_dataset = TinyImagenet(root='/home/lilipan/ling/TINYIMG', train=False, download=False, transform=trans)
        task_num = 10
        task_class = 20    
    
    for i in range(task_num):
        new_test_data = [(x, y) for x, y in test_dataset if y in range((i+1)*task_class)] 
        test_loader = torch.utils.data.DataLoader(new_test_data, batch_size=128, shuffle=False, num_workers=4)
        
        model = dataset.get_backbone()
        # if args.model == 'foster':
            # model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        model_path = f'data/results/{args.dataset}/{args.model}/TASK_{i}.pth'
        checkpoint = torch.load(model_path)
        model.load_state_dict({k[4:]:v for k,v in checkpoint.items() if "net" in k}, strict=False)
        # model.load_state_dict({k:v for k,v in checkpoint.items()}, strict=False)
        model.to(args.device)
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            print('Task:{} Accuracy: {}/{} ({:.0f}%)'.format(i,correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
            
        
    

if __name__ == '__main__':
    # test_der()
    args = get_shap_args()
    main(args)

