import numpy as np
from tools import normalization, mask_array, IoU, FC
import pickle
from tqdm import tqdm
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train model with specified dataset')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--dataset', required=True, help='Dataset name')

    return parser.parse_args()

def SHAPC_calculation(models, dataset):
    if dataset == 'seq-cifar100':
        Total_class_num = 100
        Total_task = 10
    elif dataset == 'seq-tinyimg':
        Total_class_num = 200
        Total_task = 10
    else :
        Total_class_num = 10
        Total_task = 5
    Class_per_task = Total_class_num/Total_task
    
    for model_name in models:     
        print(f'model:{model_name}')
        Mean_SHAPC = 0
        Std_SHAPC = 0
        Task_1_class_SHAPC = []
        for class_num in tqdm(range(int(Total_class_num-Class_per_task))):
            Mean_task_SHAPC = 0
            Std_task_SHAPC = 0
            pbar = tqdm(range(int(class_num/Class_per_task), Total_task), leave=False)
            for i in pbar:
                path = f'./shapley_value_conv/{dataset}/{model_name}/model{i}/class{class_num}_shap_values.npy'
                shap_values_array = np.load(path)
                if len(shap_values_array.shape) == 4:
                    shap_values_array = np.expand_dims(shap_values_array,0)
                shap_values_array = shap_values_array[[0]]
                one_channel_shap_values = np.sum(shap_values_array,axis=2)  #sum up channel dimension
                mask_shap_values = one_channel_shap_values.copy()  #keep width and height
                norm_shap = normalization(one_channel_shap_values,axis=(2,3), rank='max-min')
                for k in range(mask_shap_values.shape[1]):
                    mask_shap_values[0,k]=mask_array(mask_shap_values[0,k],0.3)
                
                if i == int(class_num/Class_per_task):
                    base_mask = mask_shap_values.astype(int)
                    base_shap = norm_shap#shap normalization
                else:
                    Class_SHAPC = np.array([])
                    mask_shap_values = mask_shap_values.astype(int)
                    for k in range(mask_shap_values.shape[1]):
                        for j in range(mask_shap_values.shape[0]):
                            _, int_mask, uni_mask = IoU(base_mask[j,k],mask_shap_values[j,k])  #width x height
                            
                            sample_SHAPC = FC(norm_shap[j,k], base_shap[j,k], int_mask, uni_mask)

                            Class_SHAPC = np.append(Class_SHAPC, sample_SHAPC)
                    Mean_class_SHAPC = np.mean(Class_SHAPC)
                    Std_class_SHAPC = np.std(Class_SHAPC)
                    #Class mean-SHAPC
                    if int(class_num/Class_per_task) == 0:
                        Task_1_class_SHAPC.extend([Mean_class_SHAPC])

                    # print(f'class{class_num} model{i} SHAPC:{Class_SHAPC:.3}')
                    Mean_task_SHAPC += Mean_class_SHAPC/(Total_task-1-int(class_num/Class_per_task))
                    Std_task_SHAPC += Std_class_SHAPC/(Total_task-1-int(class_num/Class_per_task))
            # print(f'Mean_Class_SHAPC:{Mean_Class_SHAPC:.3}')
            # print('---------------')
            Mean_SHAPC += Mean_task_SHAPC/(Total_class_num-Class_per_task)
            Std_SHAPC += Std_task_SHAPC/(Total_class_num-Class_per_task)
            Var_SHAPC = Std_SHAPC/Mean_SHAPC
        print(f'model:{model_name} Mean_SHAPC:{Mean_SHAPC*100:.3f}% Var_SHAPC:{Var_SHAPC*100:.3f}%')

            
if __name__ == '__main__':
    args = parse_args()  
    models = [args.model]  
    dataset = args.dataset
    SHAPC_calculation(models, dataset)

