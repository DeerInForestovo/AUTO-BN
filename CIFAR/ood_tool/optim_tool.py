import torch
import config as cfg
if cfg.in_dataset in ['CIFAR-10','CIFAR-100']:
    lr_c=0.01
else: lr_c=0.1

def part_opti(model):
    weight_params = []
    for pname, p in model.named_parameters():
        # print(pname)
        if (cfg.opti_part in pname):# or (cfg.opti_part2 in pname):
            weight_params +=[p]
            print('optim_tool: opti_part = ', cfg.opti_part, ', pname = ', pname)
        #if (cfg.opti_part2 in pname):
            #weight_params +=[p]
            #print(pname)
    optimizer = torch.optim.SGD([
        {'params': weight_params, 'weight_decay': 0.}],
        lr=lr_c,
        momentum=0.,
    )
    print('Optimize part: {}    Learning rate: {}'.format(cfg.opti_part,lr_c))
    
    return optimizer