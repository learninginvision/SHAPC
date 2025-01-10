CUDA_VISIBLE_DEVICES=0 python train.py --config configs/icarl_cifar10.yaml
CUDA_VISIBLE_DEVICES=0 python shap_conv_cal.py --model icarl --dataset seq-cifar10