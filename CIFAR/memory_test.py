from utils.memory_tool import memory_gen,memory_rand_gen
import torch,os
from model_pre.model_loader import get_model
import torch.backends.cudnn as cudnn
from utils.online_train_mem import online_training,online_training_odin
from utils.online_train_mem_bn import online_training_bn
from seed import set_seed
from data_pre.online_loader import set2loader,id_test_loader
from utils.threshold_tool import threshold_tool
from utils.print_results import get_and_print_results
from model_pre.trans_model import pre_model
import config as cfg
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torch.optim.lr_scheduler import StepLR
set_seed(cfg.seed)

# os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

train_loader,val_loader,in_out_dataloader,in_clas=set2loader(cfg.in_dataset,cfg.out_dataset,cfg.val_dataset)
test_loader = id_test_loader(in_dataset=cfg.in_dataset)
#if cfg.in_dataset=='CIFAR-10':
    #mem_bank,label_mem=memory_rand_gen(train_loader,10)
#else:
mem_bank,label_mem=memory_gen(train_loader,in_clas)
if cfg.in_dataset in ['CIFAR-10','CIFAR-100']:
    load = cfg.trained_classification
    model1=get_model(in_clas,False,cfg.in_dataset,cfg.model_arch)
    model2=get_model(in_clas,False,cfg.in_dataset,cfg.model_arch)
    # state_list = torch.load('../test/saves/main0 copy.pth', map_location='cpu')

    # def special_resnet(n_class=10):
    #     global device
    #     main = tv.models.resnet34()
    #     main.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    #     main.fc = torch.nn.Linear(512, n_class)
    #     main.to(device)
    #     main.eval()
    #     return main

    # model1 = special_resnet()
    # model1.load_state_dict(state_list)
    # model2 = special_resnet()
    # model2.load_state_dict(state_list)
    if not load:
        # 在这里训练CIFAR分类模型。在最终的版本中没有用到。
        print("Train on CIFAR.")
        learning_rate = 0.001
        weight_decay = 1e-4
        epochs = 20
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model1 = model1.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model1.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            model1.train()
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model1(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    tepoch.set_postfix(loss=running_loss/(tepoch.n+1), accuracy=100.*correct/total)
            model1.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model1(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f'Accuracy of the model on the test dataset: {100 * correct / total:.2f}%')
            if (epoch + 1) % 5 == 0:
                state = {'net': model1.state_dict()}
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                if cfg.online_train:
                    torch.save(state, './checkpoint/temp_classification_'+cfg.in_dataset+'_'+cfg.model_arch+'.pth.tar')

        state = {'net': model1.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if cfg.online_train:
            torch.save(state, './checkpoint/classification_'+cfg.in_dataset+'_'+cfg.model_arch+'.pth.tar')
        model2.load_state_dict(model1.state_dict())
        print("Have trained on CIFAR.")

else:  # ImageNet
    model=pre_model(cfg.model_pretrain)
    model.cuda()
#threshold_tool(model1,test_loader,device)
mean_th,std_th= 0.995, 0.05 #threshold_tool(model1,train_loader,device)

print("model part name list:")
print([pname for pname, p in model1.named_parameters()])

if cfg.ood_score in ['msp','energy']:
    online_training_alogrithm = online_training_bn if cfg.use_bn else online_training
    in_score, out_score = online_training_alogrithm(cfg.ood_score,model1,model2,in_out_dataloader,device,mean_th,std_th,mem_bank,label_mem,cfg.T_me)
    print('ID Dataset: ', cfg.in_dataset, len(in_score), ' samples')
    print('OOD Dataset: ', cfg.out_dataset, len(out_score), ' samples')
    print('opti_part: ', cfg.opti_part)
    print('ood_score: ', cfg.ood_score)
    get_and_print_results(in_score, out_score)
elif cfg.ood_score =='odin':
    in_score,out_score=online_training_odin(cfg.ood_score,model1,model2,in_out_dataloader,device,mean_th,std_th,mem_bank,label_mem,cfg.T,cfg.noise)
    get_and_print_results(in_score, out_score)

