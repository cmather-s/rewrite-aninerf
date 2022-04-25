import torch
import torch.optim

def make_optimizer(cfg, net, lr=None, weight_decay=None):
    params = []
    lr = cfg.train.lr if lr is None else lr
    weight_decay = cfg.train.weight_decay if weight_decay is None else weight_decay

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # 先用Adam
    optimizer = torch.optim.Adam(params, lr, momentum=0.9)

    return optimizer
