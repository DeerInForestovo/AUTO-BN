import torch
import config as cfg
def get_model(num_classes, load_ckpt=False,dataset='',model_arch=''):
    if dataset == 'imagenet':
        if model_arch == 'resnet18':
            from model_pre.resnet import resnet18
            model = resnet18(num_classes=num_classes, pretrained=True)
        elif model_arch == 'resnet50':
            from model_pre.resnet import resnet50
            model = resnet50(num_classes=num_classes, pretrained=True)
        elif model_arch == 'mobilenet':
            from model_pre.mobilenet import mobilenet_v2
            model = mobilenet_v2(num_classes=num_classes, pretrained=True)
    else:
        if model_arch == 'resnet18':
            from model_pre.resnet import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes)
        elif model_arch in ['resnet34','resnet34_logitnorm']:
            model = load_resnet34(dataset)
        elif model_arch == 'resnet50':
            from model_pre.resnet import resnet50
            model = resnet50(num_classes=num_classes)
        elif model_arch=='wrn':
            from model_pre.wrn import wrn40_2
            model = wrn40_2(num_classes=num_classes)
        elif model_arch=='densenet':
            from model_pre.densenet import DenseNet3
            model = DenseNet3(100, num_classes, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=None)
        else:
            assert False, 'Not supported model arch: {}'.format(model_arch)

        if load_ckpt:
            ckpt_name=cfg.ckpt_name
            print(ckpt_name)
            checkpoint = torch.load(ckpt_name)
            model.load_state_dict(checkpoint['net'])
    model.cuda()
    model.eval()
    # get the number of model parameters
    #print('Number of model parameters: {}'.format(
        #sum([p.data.nelement() for p in model.parameters()])))
    return model


def load_resnet34(dataset):
    # original ver0:
    # from model_pre.resnet import resnet34_cifar
    # model = resnet34_cifar(num_classes=num_classes)

    # my ver1, resnet 110:
    # from model_pre.cifar10_resnet.resnet import resnet110
    # model = resnet110()
    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load('./model_pre/cifar10_resnet/pretrained_models/resnet110-1d1ed7c2.th')['state_dict'])

    # my ver2, timm, resnet 34:
    # import timm
    # model = timm.create_model("hf_hub:edadaltocg/resnet34_cifar10", pretrained=True)
    # model.eval()

    # my ver2.1, offlined ver2:
    import timm
    import os

    if dataset == 'CIFAR-10':
        model = timm.create_model(
            "resnet34",
            pretrained=False,
            num_classes=10,
        )

        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "resnet34_cifar10", "resnet34_cifar10.pth")
        model.load_state_dict(torch.load(model_path))
    
        return model
    else:  # CIFAR-100
        model = timm.create_model(
            "resnet34",
            pretrained=False,
            num_classes=100
        )

        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "resnet34_cifar100", "resnet34_cifar100.pth")
        model.load_state_dict(torch.load(model_path))
    
        return model