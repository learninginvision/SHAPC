import os
import numpy as np
import torch, torchvision
from torch.utils import data
from torchvision import transforms, datasets
from tools import normalization, mask_array
from plot_tool import image as image_plot
from tqdm import tqdm
import json
from shap.plots import colors
from datasets.seq_tinyimagenet import TinyImagenet

mask = True

trans = transforms.Compose([transforms.ToTensor()])
batch_size = 50
data_test = TinyImagenet(root='data/TINYIMG',train=False, transform=trans, download=False)

def visualize_fig(model_name, dataset_name, class_idx_list, chosen_sample_number, mask):
    for class_num in class_idx_list:
        ##load initial image
        new_test_data = [(x, y) for x, y in data_test if y in [class_num]] 
        test_iter = data.DataLoader(new_test_data, batch_size, shuffle=False)
        batch = next(iter(test_iter))
        X, y = batch
        X_numpy = X.numpy()
        X_numpy = np.swapaxes(np.swapaxes(X_numpy, 1, 2), 2, 3)
        to_explain = X_numpy[[chosen_sample_number]]
        
        ##load shapley value
        for i in tqdm(range(int(class_num/20), 10)):  # class_num  -->  task_id 
            path = f'./shapley_value_conv/{dataset_name}/{model_name}/model{i}/class{class_num}_shap_values.npy'
            shap_values_array = np.load(path)
            shap_list = []
            if len(shap_values_array.shape) == 5:
                for j in range(shap_values_array.shape[0]):
                    shap_list.append(shap_values_array[j])
            else:
                for x in range(2):
                    shap_list.append(shap_values_array)   #（class x samples x channels x width x height）
            shap_list = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_list] ##（samples x width x height x channels）
            shap_list = [s[[chosen_sample_number]] for s in shap_list]
            
            if class_num % 2 == 1:
                indexes = [[class_num, class_num-1]]
            else:
                indexes = [[class_num, class_num+1]]
            index_names = np.vectorize(lambda x: str(x))(indexes)
            

            shap_list = [normalization(np.sum(s,axis=3,keepdims=True), axis=(1,2), rank='max-min') for s in shap_list]        
            os.makedirs(f'./fig/{dataset_name}/norm/{model_name}/sample{chosen_sample_number}/class{class_num}',exist_ok=True)
            save_path = f'./fig/{dataset_name}/norm/{model_name}/sample{chosen_sample_number}/class{class_num}/model{i}.png'

            image_plot(shap_list, to_explain, labels=index_names, show=False, save_path=save_path, custom_bar=False) #shape:（samples x width x height x channels）
            
            ##show mask or not
            if  mask:
                mask_list = []
                shap_mask = mask_array(shap_list[0].squeeze(), percent=0.3, rank='max')
                mask_list.append(shap_mask.reshape(shap_list[0].shape))
                shap_mask = mask_array(shap_list[1].squeeze(), percent=0.3, rank='min')
                mask_list.append(shap_mask.reshape(shap_list[0].shape))
                                        
                os.makedirs(f'./fig/{dataset_name}/mask/{model_name}/sample{chosen_sample_number}/class{class_num}',exist_ok=True)
                mask_save_path = f'./fig/{dataset_name}/mask/{model_name}/sample{chosen_sample_number}/class{class_num}/model{i}.png'
                image_plot(shap_list, to_explain, labels=index_names, show=False, save_path=mask_save_path, custom_bar=False, shapley_mask=mask_list)
        # print()

if '__main__' == __name__:
    # for model_name in (['lwf','si','icarl','agem','der','derpp','bfp']):
    dataset_name = 'seq-tinyimg'
    class_idx_list = range(10)
    for model_name in (['icarl']):
        for chosen_sample_number in range(10):
            print(f'model:{model_name}, sample number:{chosen_sample_number}')
            visualize_fig(model_name, dataset_name, class_idx_list, chosen_sample_number, mask)