import torch
import torch.nn as nn
import torch.optim as optim


# BPR learning for MF model
class BPRModel(nn.Module):

    def __init__(self, user_num, item_num, latent_factors, learning_rate, reg, mean, stdev):
        '''
        :param user_num: the number of users
        :param item_num: the number of items
        :param latent_factors: the latent factors for matrix factorization
        :param learning_rate: the learning rate for the model
        :param reg: the model regularization rate
        :param mean: the mean of a normal distribution
        :param stdev: standard deviation of a normal distribution
        '''
        super(BPRModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_factors = latent_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.mean = mean
        self.stdev = stdev

        # user & item latent vectors
        self.U = torch.normal(mean=self.mean * torch.ones(self.user_num, self.latent_factors), std=self.stdev).requires_grad_()
        # print(self.U.shape)
        self.V = torch.normal(mean=self.mean * torch.ones(self.item_num, self.latent_factors), std=self.stdev).requires_grad_()

        # define optimizer
        self.mf_optim = optim.Adam([self.U, self.V], lr=self.learning_rate)


    def forward(self, u, i, j):
        '''
        define the BPR model and complete the forward propagation process,
        In the training process, the posterior probability is obtained according to the formula,
        and then the derivative is calculated to update the values of the two matrices.
        :param u: user id.
        :param i: positive item id.
        :param j: negative item id.
        :r_ui: predicted score between user and positive item.
        :r_uj: predicted score between user and negative item.
        :return:
        :loss: BPR loss.
        '''
        u = u.long()
        i = i.long()
        j = j.long()
        r_ui = torch.diag(torch.mm(self.U[u], self.V[i].t()))
        r_uj = torch.diag(torch.mm(self.U[u], self.V[j].t()))
        regular = self.reg * (torch.sum(self.U[u] ** 2) + torch.sum(self.V[i] ** 2) + torch.sum(self.V[j] ** 2))
        loss = regular - torch.sum(torch.log2(torch.sigmoid(r_ui - r_uj)))
        return loss




