
#test
seed=1
gpu='5'

#是否将OOD样本作为正样本
out_as_pos=False
optimal_t=False

#ODIN
T=1000
noise=0.0014

online_train=True
#dataset
test_bs=16
opti_part='layer4' #WRN:block3  ResNet:layer4
# opti_part2='layer4'

use_bn=False  # 是否使用bn加权的版本
bn_section='layer2'  # layer1 layer2 layer3 layer4, 或者空串（使用所有层，不推荐）
bn_tau=1.5
bn_alpha=0.05
bn_gamma=0.5


in_dataset="CIFAR-10"   #"CIFAR-10"  "CIFAR-100"  "Imagenet"
out_dataset='SVHN' #'SVHN','Textures','LSUN_crop','LSUN_resize','places365','iSUN'

trained_classification = True  # have trained a classification one or not

val_dataset='LSUN_resize'
val_perc=0.3

ood_score='energy' # msp odin energy
train_time=2 #T
in_weight=1.
ood_weight=1. #lambda_1
consis_weight=0.1 #lambda_2
consis_idx=0.2 #phi
model_arch='resnet34' #model arch
hyperpara_out=3. #k_2
hyperpara_in=0. #k_1

if in_dataset=="CIFAR-10":
    out_datasets=['SVHN','Textures','LSUN_crop','LSUN_resize','places365','iSUN']
    if model_arch=='resnet34_logitnorm':
        ckpt_name='/data/online_ood/ckpt/cifar10/resnet34_SC/logit_norm_cos_anneal_0.1_0_0.04.pth.tar'
    if model_arch=='resnet34':
        ckpt_name='/data/online_ood/ckpt/cifar10/resnet34_SC/org_multi_step_0.1_0_1.0.pth.tar'
        # ckpt_name='./checkpoint/classification_'+in_dataset+'_'+model_arch+'.pth.tar'
    if model_arch=='wrn':
        ckpt_name='/data/6_ood_sta/checkpoint/cifar10_wrn.pth.tar'
        # ckpt_name='/data/online_ood/ckpt/cifar10/resnet18_SC/org_multi_step_0.1_0.pth.tar'
    if (model_arch=='resnet34_logitnorm') & (ood_score=='energy'):
        T_me=0.1
    else:
        T_me=1.0
elif in_dataset=="CIFAR-100":
    out_datasets = ['SVHN', 'Textures', 'LSUN_crop', 'LSUN_resize', 'places365', 'iSUN']
    if model_arch=='resnet18':
        ckpt_name='/data/online_ood/ckpt/cifar100/resnet18_SC/org_multi_step_0.1_0.pth.tar'
    if model_arch=='resnet34':
        ckpt_name='/data/online_ood/ckpt/cifar100/resnet34_SC/org_multi_step_0.1_0_1.0.pth.tar'
        # ckpt_name='./checkpoint/classification_'+in_dataset+'_'+model_arch+'.pth.tar'
    if model_arch=='resnet34_logitnorm':
        ckpt_name='/data/online_ood/ckpt/cifar100/resnet34_SC/logit_norm_multi_step_0.1_0_0.05.pth.tar'
    if model_arch=='wrn':
        ckpt_name='/data/6_ood_sta/checkpoint/cifar100_wrn.pth.tar'
    if (model_arch=='resnet34_logitnorm') & (ood_score=='energy'):
        T_me=0.01
    else:
        T_me=1.0
elif in_dataset=="domain_real":
    out_datasets = ['realb','quickdrawa','quickdrawb','sketcha','sketchb','infographa','infographb']
elif in_dataset=="Imagenet":
    out_datasets = ['Textures','iNaturalist','Places50','SUN']
    model_base=['Vit_b_16','Vit_l_16','Bit_1','Bit_3','Swin_b','Swin_l']
