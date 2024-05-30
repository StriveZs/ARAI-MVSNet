import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Distance
# -----------------------------------------------------------------------------

def pdist(feature):
    square_sum = torch.sum(feature ** 2, 1, keepdim=True)
    square_sum = square_sum + square_sum.transpose(1, 2)
    distance = torch.baddbmm(square_sum, feature.transpose(1, 2), feature, alpha=-2.0)
    return distance


def pdist2(feature1, feature2):
    square_sum1 = torch.sum(feature1 ** 2, 1, keepdim=True)
    square_sum2 = torch.sum(feature2 ** 2, 1, keepdim=True)
    square_sum = square_sum1.transpose(1, 2) + square_sum2
    distance = torch.baddbmm(square_sum, feature1.transpose(1, 2), feature2, alpha=-2.0)
    return distance


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def encode_one_hot(target, num_classes):
    one_hot = target.new_zeros(target.size(0), num_classes)
    one_hot = one_hot.scatter(1, target.unsqueeze(1), 1)
    return one_hot.float()


def smooth_cross_entropy(input, target, label_smoothing):
    assert input.dim() == 2 and target.dim() == 1
    assert isinstance(label_smoothing, float)
    batch_size, num_classes = input.shape
    one_hot = torch.zeros_like(input).scatter(1, target.unsqueeze(1), 1)
    smooth_one_hot = one_hot * (1 - label_smoothing) + torch.ones_like(input) * (label_smoothing / num_classes)
    log_prob = F.log_softmax(input, dim=1)
    loss = (- smooth_one_hot * log_prob).sum(1).mean()
    return loss


