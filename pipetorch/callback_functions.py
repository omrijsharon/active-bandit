import torch


def accuracy(output, target):
    return (torch.argmax(output, dim=1)==target).float().mean()


def recall_precision_f1(output, target):
    num_classes = output.size(1)
    predicted_classes = torch.argmax(output, dim=1)
    target_true = 0
    predicted_true = 0
    correct_true = 0
    for i in range(num_classes):
        target_true += (target == i).sum().item()
        predicted_true += (predicted_classes == i).sum().item()
        correct_true += ((predicted_classes == target)*(predicted_classes == i)).sum().item()
    recall = correct_true / target_true
    precision = correct_true / predicted_true
    f1_score = 2 * precision * recall / (precision + recall)
    return recall, precision, f1_score


def grad_abs_mean(model):
    grad_abs_sum = 0
    num_param = 0
    for param in model.parameters():
        grad_abs_sum += param.grad.abs().sum().item()
        num_param += torch.prod(torch.tensor(param.grad.size())).item()
    return grad_abs_sum / num_param