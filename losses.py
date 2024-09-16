import torch
from torch import nn
from torch.nn import functional as F


class SupConLoss(
    nn.Module
):  # from : https://github.com/ilyassmoummad/scl_icbhi2017/blob/main/losses.py
    def __init__(
        self, temperature=0.06, device="cuda:0"
    ):  # temperature was not explored for this task
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, projection1, projection2, labels=None):
        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        features = torch.cat(
            [projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1
        )
        features_acl = torch.cat([projection1, projection2], dim=0)
        batch_size = features.shape[0]

        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        print(torch.equal(contrast_feature, features_acl))

        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T), self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # for numerical stability

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device),
            0,
        )
        # or simply : logits_mask = torch.ones_like(mask) - torch.eye(50)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss1 = -mean_log_prob_pos
        loss1 = loss1.view(contrast_count, batch_size).mean()

        loss2 = amc(self.device, contrast_feature)

        alpha = 0.75

        loss = alpha * loss1 + (1 - alpha) * loss2

        return loss


## TODO: Implement another class for angular contrastive loss (ACL) as proposed by Shanshan
class AngConLoss(nn.Module):
    def __init__(self, temperature=0.06, device="cuda:0"):
        super().__init__()
        self.temperature = temperature
        self.device = device

    ## TODO: look at Shanshan's implementation and adjust this
    def forward(self, projection1, projection2, labels=None):
        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        features = torch.cat(
            [projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1
        )
        batch_size = features.shape[0]

        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T), self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # for numerical stability

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device),
            0,
        )
        # or simply : logits_mask = torch.ones_like(mask) - torch.eye(50)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss1 = -mean_log_prob_pos
        loss1 = loss1.view(contrast_count, batch_size).mean()

        loss2 = amc(self.device, contrast_feature)

        alpha = 0.75

        loss = alpha * loss1 + (1 - alpha) * loss2

        return loss


def amc(device, features):
    bs = features.shape[0] / 2

    labels = torch.cat([torch.arange(bs) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    # print(features.shape)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # logits = torch.cat([positives, negatives], dim=1)

    # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

    # logits = logits / self.args.temperature

    # if S_ij = 0
    m = 0.5
    negatives = torch.clamp(negatives, min=-1 + 1e-10, max=1 - 1e-10)
    clip = torch.acos(negatives)
    # cos^-1 (hi, hj)
    b1 = m - clip
    # mg - cos^-1(hi, hj)
    mask = b1 > 0
    l1 = torch.sum((mask * b1) ** 2)

    # if S_ij = 1
    positives = torch.clamp(positives, min=-1 + 1e-10, max=1 - 1e-10)
    l2 = torch.acos(positives)
    l2 = torch.sum(l2**2)
    # __import__("pdb").set_trace()

    loss = (l1 + l2) / 25
    # print(l1,l2,l)
    # __import__("pdb").set_trace()

    return loss
