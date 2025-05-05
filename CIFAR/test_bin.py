import torch.nn.functional
from ood_tool.optim_tool import part_opti
from utils.memory_tool import memory_gen,memory_rand_gen
import torch,os
from model_pre.model_loader import get_model
import torch.backends.cudnn as cudnn
from utils.online_train_mem import online_training,online_training_odin
from seed import set_seed
from data_pre.online_loader import set2loader,id_test_loader
from utils.print_results import get_and_print_results
from model_pre.trans_model import pre_model
import config as cfg
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
set_seed(cfg.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

train_loader,val_loader,in_out_dataloader,in_clas=set2loader(cfg.in_dataset,cfg.out_dataset,cfg.val_dataset)
test_loader = id_test_loader(in_dataset=cfg.in_dataset)
#if cfg.in_dataset=='CIFAR-10':
    #mem_bank,label_mem=memory_rand_gen(train_loader,10)
#else:
bit_num = 256
class Main(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.main = tv.models.resnet34()
        self.main.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.main.fc = torch.nn.Linear(512, bit_num)

        self.register_buffer('bits', torch.tensor([torch.randperm(n_class).tolist() for _ in range(bit_num)]).T<n_class//2)
        self.bits

mem_bank,label_mem=memory_gen(train_loader,in_clas)
if cfg.in_dataset in ['CIFAR-10','CIFAR-100']:
    load = cfg.trained_classification
    # model1=get_model(in_clas,False,'',cfg.model_arch)
    # model2=get_model(in_clas,False,'',cfg.model_arch)
    state_list = torch.load('../test/saves/main1 copy.pth', map_location='cpu')

    def special_resnet(n_class=10):
        main = Main(n_class)
        # main.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # main.fc = torch.nn.Linear(512, n_class)
        main.to('cuda')
        main.eval()
        return main

    model1 = special_resnet()
    model1.load_state_dict(state_list)
    bits = model1.bits
    model1 = model1.main

    model2 = special_resnet()
    model2.load_state_dict(state_list)
    model2 = model2.main


else:
    model=pre_model(cfg.model_pretrain)
    model.cuda()
# threshold_tool(model1,test_loader,device)
def threshold_tool(net,loader,device):
    score=[]
    net.eval()
    for batch_idx, (image, label) in enumerate(loader):
        image = image.to(device)
        logits = net(image)
        score.append(logits.abs().sigmoid().mean(dim=-1))

    score=torch.cat(score)

    mean=score.mean()
    delta=score.std()
    print('Mean: {}    Std: {}'.format(mean,delta))
    return mean.item(),delta.item()
with torch.no_grad():
    # threshold_tool(model1,test_loader,device)
    mean_th,std_th= 0.9829138517379761, 0.01488920021802187#threshold_tool(model1,train_loader,device)


def batch_train(mem_b,mem_l,net,net2,input,optim,out=0):  # out=1 认为ID
    to_np = lambda x: x.data.cpu().numpy()
    x1=0
    while x1<cfg.train_time:
        optim.zero_grad()
    
        if cfg.in_dataset == 'Imagenet':
            loss_func1 = nn.CrossEntropyLoss(reduction='sum')
        else:
            loss_func1 = nn.CrossEntropyLoss()
        
        #if cfg.in_dataset=='Imagenet':

        #    first_idx=True
        #    for i in range (len(mem_b)):
        #        logit_in = net(mem_b[i].cuda())
        #        mem_b[i].cpu()
        #        if first_idx==True:
        #            loss1 = loss_func1(logit_in, mem_l[i].cuda())
        #        else: loss1 += loss_func1(logit_in, mem_l[i].cuda())
        #        mem_l[i].cpu()
        #    loss1=loss1/1000.

        #else:
        #    logit_in = net(mem_b.cuda())
        #    loss1 = loss_func1(logit_in, mem_l.cuda())

        # loss1 = loss_func1(net(input.cuda()),mem_l.cuda())  # ?
        pre = net(mem_b.cuda())
        loss1 = torch.where(bits[mem_l.cuda()], nn.functional.softplus(pre), nn.functional.softplus(-pre)).mean()
        #loss_func2 =nn.BCEWithLogitsLoss()
        logit_out = net(input)
        with torch.no_grad():
            logit_out_org=net2(input)
        # smax_out_org = to_np(torch.nn.functional.softmax(logit_out_org, dim=1))
        # pred_out_org = np.argmax(smax_out_org, axis=1)
        loss2 = logit_out.abs().sigmoid().mean()
        loss3 = torch.where((logit_out_org*logit_out)>0, torch.zeros_like(logit_out), 2*logit_out.abs().sigmoid()).mean()
        #print(loss1,loss2,loss3)
        #loss=cfg.in_weight*loss1+cfg.ood_weight*loss2#+0.2*loss3
        if out==0:  # OOD sample
            loss=cfg.ood_weight*loss2+cfg.in_weight*loss1+cfg.consis_weight*loss3
        else:  # ID sample
            loss=cfg.in_weight*loss1
        loss.backward()
        optim.step()
        x1+=1

def online_training(score,net,net2,loader,device,mean,std,mem_bank,mem_label,Tem=1.0):
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
    mem_idx=0
    print(score)
    print('In_weight:  {}      Out_weight:{}    Consis_weight:{}'.format(cfg.in_weight,cfg.ood_weight,cfg.consis_weight))
    print('Consistent threshold:{}'.format(cfg.consis_idx))
    print('In_border:  {}      Out_border:{}'.format(cfg.hyperpara_in,cfg.hyperpara_out))
    print('Online Train:  {}'.format(cfg.online_train))
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    parameters = list(net.parameters())
    if cfg.opti_part == 'all':
        print('all')
        optimizer = torch.optim.SGD(parameters, lr=0.001, momentum=0., weight_decay=0.)
    else:
        optimizer = part_opti(net)

    ood_list = []
    for batch_idx, (data, dis_idx) in enumerate(loader):  # dis_idx=0 来自ID, dis_idx=1 来自OOD
        image, label = data[0], data[1]

        if batch_idx==0:
            print(batch_idx)
            in_border=mean+cfg.hyperpara_in*std
            out_border=mean-cfg.hyperpara_out*std
            border_outcount.append(out_border)

        image = image.to(device)
        logits = net(image)
        logits: torch.Tensor
        pre_max = logits.abs().sigmoid().mean().item()

        # print(dis_idx.item(), 256-((logits[:,None,:]<0)==bits).sum(dim=-1).max().item())
        if pre_max<out_border:  # 认为OOD
            # print('OOD')
            ood_list.append(image)
            out_training_count+=1
            border_outcount.append(pre_max)
            out_border = np.array(border_outcount).mean()  # 更新 m_out
            if cfg.online_train and len(ood_list)>=8:
                batch_train(mem_bank,mem_label,net,net2,torch.cat(ood_list),optimizer,out=0)
                ood_list = []
            if dis_idx==0:  # 来自ID，那么误认为OOD
                wrong_train_out+=1
            train_idx+=1

        elif pre_max>in_border:  # 认为ID
            # print('ID')
            pred=torch.max((logits[:,None,:] * (1-2*bits)).sum(dim=-1), dim=-1).indices.item()
            # print(logits.shape, bits.shape)

            if cfg.in_dataset in ['CIFAR-10','CIFAR-100']:
                for i in range(logits.shape[0]):
                    if pred==mem_label[i].item():
                        mem_bank[i]=torch.squeeze(image,dim=0)
                #if mem_idx<len(mem_label):
                    #mem_bank[mem_idx]=torch.squeeze(image,dim=0)
                    #mem_label[mem_idx]=torch.Tensor(pred)
                    #mem_idx+=1
                #else:
                    #mem_idx=0
                    #mem_bank[mem_idx] = torch.squeeze(image, dim=0)
                    #mem_label[mem_idx] = torch.Tensor(pred)
            else:
                for i in range (len(mem_label)):
                    for j in range (len(mem_label[i])):
                        if pred==mem_label[i][j].item():
                            mem_bank[i][j] = torch.squeeze(image, dim=0)
            in_training_count+=1
            # batch_train(mem_bank,torch.Tensor(pred),net,net2,image,optimizer,out=1)
            # batch_train(mem_bank,mem_label,net,net2,image,optimizer,out=1)

            if dis_idx==1:
                wrong_train_in+=1

        preds = torch.max((logits[:,None,:] * (1-2*bits)).sum(dim=-1), dim=-1).indices.item()
        label = label.item()
        all_score = to_np(-logits.square().mean(dim=-1))
        # if score == 'energy':
        #     all_score = -to_np(Tem * torch.logsumexp(logits / Tem, dim=1))
        # elif score == 'msp':
        #     all_score = -np.max(to_np(torch.nn.functional.softmax(logits / Tem, dim=1)), axis=1)

        if dis_idx == 0:  # in_distribution
            in_dis_score.append(all_score)
            if preds == label:
                right_score.append(all_score)
            else:
                wrong_score.append(all_score)
        else:  # ood ditribution
            out_dis_score.append(all_score)
        if (batch_idx%1000==0)&(batch_idx>0):
            print(batch_idx)
            working_time=time.time()-start_time
            #print("Training Time: ", working_time)
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

if cfg.ood_score in ['msp','energy']:
    in_score,out_score=online_training(cfg.ood_score,model1,model2,in_out_dataloader,device,mean_th,std_th,mem_bank,label_mem,cfg.T_me)
    print(len(in_score),len(out_score))
    get_and_print_results(in_score, out_score)
elif cfg.ood_score =='odin':
    in_score,out_score=online_training_odin(cfg.ood_score,model1,model2,in_out_dataloader,device,mean_th,std_th,mem_bank,label_mem,cfg.T,cfg.noise)
    get_and_print_results(in_score, out_score)
