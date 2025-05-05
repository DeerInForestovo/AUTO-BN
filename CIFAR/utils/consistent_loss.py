import torch.nn as nn
import torch

def get_margin_loss(pred, label,t_value):
    """
    pred: [N, C]
    """
    # print(pred)
    return torch.max(pred, dim=-1).values.mean() - torch.gather(pred, dim=-1, index=label.reshape(-1,1)).mean()

    n, c = pred.shape
    top_pred= pred[0][label]
    d_pred = top_pred.repeat(c, 1).t() - pred
    label=torch.tensor(label)
    #print(label)
    torch.wh
    temp = torch.zeros(pred.size()).scatter(1, label.unsqueeze(1), ).sum(dim=-1)
    d_pred += temp
    margin, _ = torch.min(d_pred, 1)
    margin_loss = torch.mean(torch.maximum(t_value - margin, torch.zeros(margin.size())))
    margin_loss = margin_loss.requires_grad_()
    return margin_loss

# def get_margin_loss(pred, label,t_value):
#     """
#     pred: [N, C]
#     """
#     n, c = pred.shape
#     top_pred= pred[0][label]
#     d_pred = top_pred.repeat(c, 1).t() - pred
#     label=torch.tensor(label)
#     #print(label)
#     temp = torch.zeros(pred.size()).scatter_(1, label.unsqueeze(1), 1)
#     d_pred += temp
#     margin, _ = torch.min(d_pred, 1)
#     margin_loss = torch.mean(torch.maximum(t_value - margin, torch.zeros(margin.size())))
#     margin_loss = margin_loss.requires_grad_()
#     return margin_loss
