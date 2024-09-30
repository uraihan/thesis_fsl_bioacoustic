import torch
from torch import nn
from torch.nn import functional as F

import args


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
        batch_size = features.shape[0]

        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = (
                torch.eq(labels, labels.T).float().to(self.device)
            )  # is this mask of the label?

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(torch.equal(contrast_feature, features_acl))

        # NOTE: forming (exp(zi * zj) / t)
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T), self.temperature
        )

        # NOTE: what is this logits_max for?? -> this is basically taking max value for each row after multiplying features (concatenation of proj1 and proj2)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # for numerical stability

        mask = mask.repeat(contrast_count, contrast_count)
        label_mask = mask
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device),
            0,
        )
        # print(mask)
        # print(logits_mask)
        # or simply : logits_mask = torch.ones_like(mask) - torch.eye(50)
        # NOTE: what is this mask for?
        # -> is this mask the lower part of the equation?
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss1 = -mean_log_prob_pos
        loss1 = loss1.view(contrast_count, batch_size).mean()

        if args.args.loss == "scl-orig":
            return loss1

        loss2 = amc(self.device, contrast_feature, labels)
        # loss2 = amc(self.device, contrast_feature)

        alpha = 0.5

        loss = alpha * loss1 + (1 - alpha) * loss2

        return loss2


def amc(device, features, labels):
    # def amc(device, features):

    # labels = torch.eq(labels, labels).float().to(device)
    # labels = F.one_hot(labels.squeeze(1)).float().to(device)
    # labels = labels.repeat(2, 2)

    # features = F.normalize(features, dim=1)
    # print(features.shape)
    # mask = torch.zeros(labels.shape[0], labels.shape[0])
    labels = labels.repeat(2, 1)
    # label_mask = [
    #     1 if labels[i] == labels[j] else 0
    #     for j in range(len(labels))
    #     for i in range(len(labels))
    # ]
    # label_mask = torch.FloatTensor(label_mask).to(device).reshape(features.shape[0], -1)
    label_mask = torch.eq(labels, labels.T)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    diag_mask = torch.eye(label_mask.shape[0], dtype=torch.bool).to(device)
    label_mask = label_mask[~diag_mask].view(
        label_mask.shape[0], -1
    )  # Q: what is the purpose of this?
    similarity_matrix = similarity_matrix[~diag_mask].view(
        similarity_matrix.shape[0], -1
    )

    # select and combine multiple positives
    # WARN: each row has different length
    # dap> print(torch.sum(labels[0]))
    # tensor(11.)
    #
    # dap> print(torch.sum(labels[1]))
    # tensor(13.)
    #
    # dap> print(torch.sum(labels[2]))
    # tensor(3.)
    #
    # dap> print(torch.sum(labels[3]))
    # tensor(15.)
    #
    # dap> print(torch.sum(labels[4]))
    # tensor(11.)
    positives = similarity_matrix[label_mask.bool()]

    # select only the negatives the negatives
    negatives = similarity_matrix[~label_mask.bool()]

    # if S_ij = 0
    m = 0.5
    negatives = torch.clamp(negatives, min=-1 + 1e-10, max=1 - 1e-10)
    clip = torch.acos(negatives)
    l1 = torch.max(torch.zeros(clip.shape[0]).to(device), (m - clip))
    l1 = torch.sum(l1**2)

    # if S_ij = 1
    positives = torch.clamp(positives, min=-1 + 1e-10, max=1 - 1e-10)
    l2 = torch.acos(positives)
    l2 = torch.sum(l2**2)

    loss = (l1 + l2) / 50

    return loss
