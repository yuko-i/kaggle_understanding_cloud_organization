import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F



class SoftTripleLoss(nn.Module):

    """Qi Qian, et al.,
    `SoftTriple Loss: Deep Metric Learning Without Triplet Sampling`,
    https://arxiv.org/abs/1909.05235
    """

    def __init__(
        self,
        embedding_dim: int=3,
        num_categories: int=4,
        use_regularizer: bool = True,
        num_initial_center: int = 2,
        similarity_margin: float = 0.1,
        coef_regularizer1: float = 1e-2,
        coef_regularizer2: float = 1e-2,
        coef_scale_softmax: float = 1.0,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """Constructor
        Args:
            embedding_dim: dimension of inputs to this module (N x embedding_dim)
            num_categories: total category count to classify
            num_initial_center: initial number of centers for each categories
            similarity_margin: margin term as is in triplet loss
            coef_regularizer1: entropy regularizer for dictibution over classes
            coef_regularizer2: regularizer for cluster variancce.
            coef_scale_softmax: scaling factor before final softmax op
            device: device on which this loss is computed
        """

        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_categories = num_categories
        self.use_regularizer = use_regularizer
        self.num_initial_center = num_initial_center
        self.delta = similarity_margin
        self.gamma_inv = 1 / coef_regularizer1
        self.tau = coef_regularizer2
        self.lambda_ = coef_scale_softmax
        self.device = device
        self.fc_hidden = nn.Linear(
            embedding_dim, num_categories * num_initial_center
        ).to(device)
        nn.init.xavier_normal_(self.fc_hidden.weight)
        self.base_loss = nn.CrossEntropyLoss().to(device)
        self.softmax = nn.Softmax(dim=2).to(device)

    def infer(self, embedding):
        weight = F.normalize(self.fc_hidden.weight)
        x = F.linear(embedding, weight).view(
            -1, self.num_categories, self.num_initial_center
        )
        x = self.softmax(x.mul(self.gamma_inv)).mul(x).sum(dim=2)
        return x

    def cluster_variance_loss(self):
        weight = F.normalize(self.fc_hidden.weight)
        loss = 0.0
        for i in range(self.num_categories):
            weight_sub = weight[
                i * self.num_initial_center : (i + 1) * self.num_initial_center
            ]
            subtraction_norm = 1.0 - torch.matmul(
                weight_sub, weight_sub.transpose(1, 0)
            )
            subtraction_norm[subtraction_norm <= 0.0] = 1e-10
            loss += torch.sqrt(2 * subtraction_norm.triu(diagonal=1)).sum()

        loss /= (
            self.num_categories
            * self.num_initial_center
            * (self.num_initial_center - 1)
        )
        return loss

    def forward(self, embeddings, labels):
        h = self.infer(embeddings)
        one_hot = torch.zeros(h.size(), device=self.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        h = h - self.delta * one_hot
        h.mul_(self.lambda_)
        clf_loss = self.base_loss(h, labels)
        if not self.use_regularizer:
            return clf_loss

        var_loss = self.cluster_variance_loss()
        return clf_loss + self.tau * var_loss



class SoftTripleLoss2(nn.Module):
    def __init__(self, embedding_size=4, n_class=4, n_center=3,
                 lmd=10.0, gamma=0.1, tau=0.2, margin=0.01):
        super(SoftTripleLoss, self).__init__()
        self._lmd = lmd
        self._inv_gamma = 1.0 / gamma
        self._tau = tau
        self._margin = margin
        self._n_class = n_class
        self._n_center = n_center
        self.fc = Parameter(torch.Tensor(n_class * n_center, embedding_size))
        nn.init.kaiming_uniform_(self.fc)
        self.softmax = nn.Softmax(dim=2)

    def infer(self, embedding):
        weight = F.normalize(self.fc, p=2, dim=1)
        x = F.linear(embedding, weight).view(-1, self._n_class, self._n_center)
        prob = self.softmax(self._inv_gamma * x)
        return (prob * x).sum(dim=2), weight

    def forward(self, embeddings, labels,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        h, w = self.infer(embeddings)
        margin_m = torch.zeros(h.shape).to(device)
        margin_m[torch.arange(0, margin_m.shape[0]), labels] = self._margin
        loss_cls = F.cross_entropy(self._lmd * (h - margin_m), labels.squeeze(-1))
        if self._tau > 0 and self._n_center > 1:
            reg = 0.0
            for i in range(self._n_class):
                w_sub = w[i * self._n_center : (i + 1) * self._n_center]
                sub_norm = 1.0 - torch.matmul(w_sub, w_sub.transpose(1, 0))
                sub_norm[sub_norm <= 0.0] = 1e-10
                reg += torch.sqrt(2 * sub_norm.triu(diagonal=1)).sum()
            reg /= self._n_class * self._n_center * (self._n_center - 1.0)
            return loss_cls + self._tau * reg
        else:
            return loss_cls