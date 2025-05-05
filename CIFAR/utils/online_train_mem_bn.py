from __future__ import print_function
import torch
import os
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils.print_results import get_and_print_results
import config as cfg
import torch.nn as nn
from ood_tool.optim_tool import part_opti
from ood_tool.odin_tool import ODIN
import time
from utils.consistent_loss import get_margin_loss
import random
from collections import defaultdict

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# --------------------- 提取BatchNorm统计量 ---------------------
def get_bn_stats(model, input_data):
    """
    提取模型所有BN层的均值与方差，拼接为特征向量
    返回形状: (2 * sum(C_l), )
    """
    bn_features = []
    hooks = []
    
    # 定义钩子函数捕获BN统计量
    def hook_fn(module, input, output):
        if isinstance(module, nn.BatchNorm2d):
            # 计算通道级均值与方差
            channel_mean = torch.mean(output, dim=[0, 2, 3])  # (C,)
            channel_var = torch.var(output, dim=[0, 2, 3], unbiased=False)  # (C,)
            bn_features.extend([channel_mean.detach(), channel_var.sqrt().detach()])
    
    # 注册钩子到所有BN层
    for name, module in model.named_modules():
        if str(cfg.bn_section) in name and isinstance(module, nn.BatchNorm2d):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # 前向传播触发钩子
    with torch.no_grad():
        model(input_data)
    
    # 移除钩子并拼接特征
    for hook in hooks:
        hook.remove()
    
    return torch.cat(bn_features, dim=0).cpu().numpy()

# --------------------- prototype 管理器 ---------------------
class PrototypeManager:
    def __init__(self, tau_new, alpha, gamma):
        self.prototypes = []  # 原型集合 [ (feat_vec, count), ... ]
        self.tau_new = tau_new  # 新建原型阈值
        self.alpha = alpha     # 滑动更新率
        self.gamma = gamma     # 权重温度系数
        self.cnt = 0
    
    def update(self, feat):
        """
        根据新特征更新原型集合
        feat: 形状 (D, ) 的特征向量
        """
        if len(self.prototypes) == 0:
            self.prototypes.append(feat)
            return
        
        # 计算与最近原型的距离
        dists = [np.linalg.norm(feat - p) for p in self.prototypes]
        min_dist = np.min(dists)
        idx = np.argmin(dists)

        # print(min_dist)
        
        if min_dist > self.tau_new:
            # 新建原型
            self.prototypes.append(feat)
        else:
            # 滑动更新
            self.prototypes[idx] = (1 - self.alpha) * self.prototypes[idx] + self.alpha * feat
        
        self.cnt += 1
        if self.cnt % 100 == 0:
            print(self.cnt, " updates, ", len(self.prototypes), " prototypes")
    
    def get_weight(self, feat):
        """ 计算样本的置信度权重 """
        dists = [np.linalg.norm(feat - p) for p in self.prototypes]
        min_dist = np.min(dists)
        return np.exp(-self.gamma * min_dist)

def batch_train(mem_b, mem_l, net, net2, input, optim, proto_manager, out=0, device='cuda'):
    x1 = 0
    while x1 < cfg.train_time:
        optim.zero_grad()
        
        loss_func1 = nn.CrossEntropyLoss()
        loss1 = loss_func1(net(mem_b.to(device)), mem_l.to(device))
        
        # pseudo OOD
        if out == 0:  
            logit_out = net(input)
            with torch.no_grad():
                logit_out_org = net2(input)
            smax_out_org = F.softmax(logit_out_org, dim=1)
            pred_out_org = torch.argmax(smax_out_org, dim=1)
            
            # 计算置信度权重
            bn_feats = [get_bn_stats(net, x.unsqueeze(0)) for x in input]  # 提取BN特征
            weights = [proto_manager.get_weight(f) for f in bn_feats]
            weights = torch.tensor(weights, device=device).view(-1, 1)
            weights = weights.detach()
            
            # 加权 loss function
            loss2 = (1.0 * -(logit_out.mean(1) - torch.logsumexp(logit_out, dim=1)) * weights).mean()
            loss3 = get_margin_loss(logit_out, pred_out_org, cfg.consis_idx)
            
            loss = cfg.ood_weight * loss2 + cfg.in_weight * loss1 + cfg.consis_weight * loss3
        else: # pseudo ID
            loss = cfg.in_weight * loss1
        
        loss.backward()
        optim.step()
        x1 += 1

def online_training_bn(score, net, net2, loader, device, mean, std, mem_bank, mem_label, Tem=1.0):
    train_idx=0
    start_time=time.time()
    net.eval()
    net2.eval()
    in_dis_score = []
    out_dis_score = []
    right_score = []
    wrong_score = []
    in_training_count = 0
    out_training_count = 0
    wrong_train_in = 0
    wrong_train_out = 0
    in_border=0
    out_border=0
    border_outcount=[]
    print(score)
    print('In_weight:  {}      Out_weight:{}    Consis_weight:{}'.format(cfg.in_weight,cfg.ood_weight,cfg.consis_weight))
    print('Consistent threshold:{}'.format(cfg.consis_idx))
    print('In_border:  {}      Out_border:{}'.format(cfg.hyperpara_in,cfg.hyperpara_out))
    print('Online Train:  {}'.format(cfg.online_train))
    to_np = lambda x: x.data.cpu().numpy()

    parameters = list(net.parameters())
    if cfg.opti_part == 'all':
        print('all')
        optimizer = torch.optim.SGD(parameters, lr=0.001, momentum=0., weight_decay=0.)
    else:
        optimizer = part_opti(net)

    ood_list = []
    proto_manager = PrototypeManager(tau_new=cfg.bn_tau, alpha=cfg.bn_alpha, gamma=cfg.bn_gamma)  # 新增原型管理器
    
    for batch_idx, (data, dis_idx) in enumerate(loader):
        image, label = data[0], data[1]

        if batch_idx == 0:
            print(batch_idx)
            in_border = mean + cfg.hyperpara_in * std
            out_border = mean - cfg.hyperpara_out * std
            border_outcount.append(out_border)

        image = image.to(device)
        logits = net(image)
        smax = to_np(F.softmax(logits, dim=1))
        pre_max = np.max(smax, axis=1)
        
        if pre_max < out_border:  # 判定为OOD
            out_training_count += 1
            border_outcount.append(pre_max)
            out_border = np.array(border_outcount).mean()  # 更新 m_out
            
            # 提取BN特征并更新原型
            bn_feat = get_bn_stats(net, image)
            proto_manager.update(bn_feat)
            
            batch_train(mem_bank, mem_label, net, net2, torch.cat([image]), optimizer, proto_manager, out=0, device=device)  # 传入proto_manager
            if dis_idx == 0:  # 来自ID，那么误认为OOD
                wrong_train_out += 1
            train_idx += 1

        elif pre_max > in_border:  # 认为ID
            pred = np.argmax(smax, axis=1)
            for i in range (len(smax)):
                if pred == mem_label[i].item():
                    mem_bank[i] = torch.squeeze(image, dim=0)
            in_training_count += 1
            if dis_idx == 1:
                wrong_train_in += 1

        preds = np.argmax(smax, axis=1)
        label = label.numpy().squeeze()
        if score == 'energy':
            all_score = -to_np(Tem * torch.logsumexp(logits / Tem, dim=1))
        elif score == 'msp':
            all_score = -np.max(to_np(F.softmax(logits / Tem, dim=1)), axis=1)

        if dis_idx == 0:  # in_distribution
            in_dis_score.append(all_score)
            if preds == label:
                right_score.append(all_score)
            else:
                wrong_score.append(all_score)
        else:  # ood ditribution
            out_dis_score.append(all_score)
        if batch_idx % 1000 == 0 and batch_idx > 0:
            print(batch_idx)
            working_time = time.time() - start_time
            print('Wrong Train Example: In-dis: {}       OOD: {}'.format(wrong_train_in, wrong_train_out))  # 误认为是ID的个数，误认为是OOD的个数
            print('Training Example Count: In-disribution: {}       OOD: {}'.format(in_training_count, out_training_count))
            print('In-distribution Accuracy: {:.2f}%'.format(100 * len(right_score) / (len(right_score) + len(wrong_score))))
            get_and_print_results(np.array(in_dis_score), np.array(out_dis_score)) 

    print("----------------------")
    print("Training Time: ", working_time)
    print('Training Example Count: In-disribution: {}       OOD: {}'.format(in_training_count, out_training_count))
    print('In-distribution Accuracy: {:.2f}%'.format(100 * len(right_score) / (len(right_score) + len(wrong_score))))
    print('Saving..')
    state = {
            'net': net.state_dict(),
            }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if cfg.online_train:
        torch.save(state, './checkpoint/after_training_'+cfg.in_dataset+'_'+cfg.out_dataset+'_'+cfg.model_arch+'.pth.tar')
    
    return np.array(in_dis_score).copy(), np.array(out_dis_score).copy()

