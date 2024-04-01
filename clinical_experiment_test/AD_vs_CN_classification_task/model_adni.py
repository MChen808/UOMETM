import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from time import time
import numpy as np
from tqdm import tqdm
from utils import adni_utils
import math
import logging
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)
adni_utils = adni_utils()

dim_z = 16


class AE_adni(nn.Module):
    def __init__(self, input_dim, left_right=0):
        super(AE_adni, self).__init__()
        nn.Module.__init__(self)
        self.name = 'AE_adni'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.kl = 0
        self.gamma = 1
        self.lam = 10
        self.input_dim = input_dim
        self.left_right = left_right

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.mu = nn.Sequential(
            nn.Linear(128, dim_z),
            nn.BatchNorm1d(dim_z),
            # nn.Tanh(),
        )

        self.logVar = nn.Sequential(
            nn.Linear(128, dim_z),
            # nn.BatchNorm1d(dim_z),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_z, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.input_dim),
            nn.ReLU()
        )

        self.X, self.Y = None, None
        self.beta, self.b, self.D = None, None, None
        self.U, self.V = None, None
        self.sigma0_2, self.sigma1_2, self.sigma2_2 = None, None, None

    def _init_mixed_effect_model(self):
        self.beta = torch.rand(size=[self.X.size()[1], dim_z], device=self.device)
        self.b = torch.normal(mean=0, std=1, size=[self.Y.size()[1], dim_z], device=self.device)
        self.U = torch.diag(torch.tensor([1 for i in range(dim_z // 2)] + [0 for i in range(dim_z - dim_z // 2)],
                                         device=self.device)).float()
        self.V = torch.eye(dim_z, device=self.device) - self.U
        self.sigma0_2, self.sigma1_2, self.sigma2_2 = 0.1, 1., 0.15
        self.D = torch.eye(self.Y.size()[1], device=self.device).float()

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2).to(self.device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std=1).to(self.device)
        if self.kl != 0:  # beta VAE
            return mu + eps * std
        else:  # regular AE
            return mu

    def encode(self, input):
        z = self.encoder(input)
        mu = self.mu(z)
        logVar = self.logVar(z)
        if self.training:
            z = self.reparametrize(mu, logVar)
        else:
            z = mu
        return mu, logVar, z

    def forward(self, image0, image1=None):
        mu, logVar, z0 = self.encode(image0)
        zu0 = torch.matmul(z0, self.U)
        zv0 = torch.matmul(z0, self.V)
        if image1 is not None:
            _, _, z1 = self.encode(image1)
            encoded = torch.matmul(z1, self.U) + zv0
            reconstructed = self.decoder(encoded)
            return reconstructed
        else:
            encoded = zu0 + zv0
            reconstructed = self.decoder(encoded)
            return reconstructed, z0, zu0, zv0, mu, logVar

    @staticmethod
    def recon_loss(input_, reconstructed):
        # recon_loss = torch.sum((reconstructed - input_) ** 2) / input_.shape[0]
        recon_loss = torch.mean((reconstructed - input_) ** 2)
        return recon_loss

    @staticmethod
    def kl_loss(mu, logVar):
        kl_divergence = 0.5 * torch.mean(-1 - logVar + mu.pow(2) + logVar.exp())
        return kl_divergence

    def train_(self, train_data_loader, test_data_loader, optimizer, num_epochs):
        self.to(self.device)
        self._init_mixed_effect_model()
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('#### Epoch {}/{} ####'.format(epoch + 1, num_epochs))

            tloss = np.array([0., 0., 0., 0.])
            nb_batches = 0

            s_tp, Z, ZU, ZV = None, None, None, None
            # for data in tqdm(adni_utils.merge_loader(train_data_loader, test_data_loader)):
            for data in tqdm(train_data_loader):
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 labels, 5 subject, 6 timepoint
                image = data[self.left_right]
                optimizer.zero_grad()

                # self-reconstruction loss
                input_ = Variable(image).to(self.device).float()
                reconstructed, z, zu, zv, mu, logVar = self.forward(input_)
                self_reconstruction_loss = self.recon_loss(input_, reconstructed)

                # kl divergence
                kl_loss = self.kl_loss(mu, logVar)

                # store Z, ZU, ZV
                subject = torch.tensor([[s for s in data[5]]], device=self.device)
                tp = torch.tensor([[tp for tp in data[6]]], device=self.device)
                st = torch.transpose(torch.cat((subject, tp), 0), 0, 1)
                if s_tp is None:
                    s_tp, Z, ZU, ZV = st, z, zu, zv
                else:
                    s_tp = torch.cat((s_tp, st), 0)
                    Z = torch.cat((Z, z), 0)
                    ZU = torch.cat((ZU, zu), 0)
                    ZV = torch.cat((ZV, zv), 0)

                # cross-reconstruction loss
                if epoch > 30:
                    baseline_age = data[3]
                    delta_age = data[2] - baseline_age
                    index0, index1 = self.generate_sample(baseline_age, delta_age)
                    image0 = image[index0]
                    image1 = image[index1]
                    if index0:
                        input0_ = Variable(image0).to(self.device).float()
                        input1_ = Variable(image1).to(self.device).float()
                        reconstructed = self.forward(input0_, input1_)
                        cross_reconstruction_loss = self.recon_loss(input0_, reconstructed)
                        recon_loss = (self_reconstruction_loss + cross_reconstruction_loss) / 2
                    else:
                        cross_reconstruction_loss = 0.
                        recon_loss = self_reconstruction_loss
                else:
                    cross_reconstruction_loss = 0.
                    recon_loss = self_reconstruction_loss

                loss = recon_loss + self.kl * kl_loss
                loss.backward()
                optimizer.step()
                tloss[0] += float(self_reconstruction_loss)
                tloss[1] += float(kl_loss)
                tloss[2] += float(cross_reconstruction_loss)
                tloss[-1] += float(loss)
                nb_batches += 1

            # comply with generative model
            # sort_index1 = s_tp[:, 1].sort()[1]
            # sorted_s_tp = s_tp[sort_index1]
            # sort_index2 = sorted_s_tp[:, 0].sort()[1]
            # Z, ZU, ZV = Z[sort_index1], ZU[sort_index1], ZV[sort_index1]
            # Z, ZU, ZV = Z[sort_index2], ZU[sort_index2], ZV[sort_index2]
            if epoch % 5 == 0:
                test_loss, ZU_test, ZV_test = self.evaluate(test_data_loader, epoch)
                print('Testing accuracy updated...')
            if epoch > 30:
                if epoch % 5 == 0:
                    self.plot_distribution(Z, title='Z')
                    self.plot_distribution(ZU, title='ZU')
                    self.plot_distribution(ZV, title='ZV')
                    print('Distribution plotting finished...')
                self.generative_parameter_update(Z, ZU, ZV)
                print('Aligning finished...')

            epoch_loss = tloss / nb_batches

            if epoch_loss[-1] <= best_loss:
                es = 0
                best_loss = epoch_loss[-1]
            else:
                es += 1

            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss}/{test_loss} take {end_time - start_time:.3} seconds\n")
        return {'train': ZU.detach(), 'test': ZU_test.detach()}, {'train': ZV.detach(), 'test': ZV_test.detach()}

    def evaluate(self, test_data_loader, epoch):
        self.to(self.device)
        self.training = False
        self.eval()
        tloss = np.array([0., 0., 0., 0.])
        nb_batches = 0

        with torch.no_grad():
            s_tp, Z, ZU, ZV = None, None, None, None
            for data in test_data_loader:
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 labels, 5 subject, 6 timepoint
                image = data[self.left_right]

                # self-reconstruction loss
                input_ = Variable(image).to(self.device).float()
                reconstructed, z, zu, zv, mu, logVar = self.forward(input_)
                self_reconstruction_loss = self.recon_loss(input_, reconstructed)

                # kl divergence
                kl_loss = self.kl_loss(mu, logVar)

                # store Z, ZU, ZV
                subject = torch.tensor([[s for s in data[5]]], device=self.device)
                tp = torch.tensor([[tp for tp in data[6]]], device=self.device)
                st = torch.transpose(torch.cat((subject, tp), 0), 0, 1)
                if s_tp is None:
                    s_tp, Z, ZU, ZV = st, z, zu, zv
                else:
                    s_tp = torch.cat((s_tp, st), 0)
                    Z = torch.cat((Z, z), 0)
                    ZU = torch.cat((ZU, zu), 0)
                    ZV = torch.cat((ZV, zv), 0)

                # cross-reconstruction loss
                if epoch > 30:
                    baseline_age = data[3]
                    delta_age = data[2] - baseline_age
                    index0, index1 = self.generate_sample(baseline_age, delta_age)
                    image0 = image[index0]
                    image1 = image[index1]
                    if index0:
                        input0_ = Variable(image0).to(self.device).float()
                        input1_ = Variable(image1).to(self.device).float()
                        reconstructed = self.forward(input0_, input1_)
                        cross_reconstruction_loss = self.recon_loss(input0_, reconstructed)
                        recon_loss = (self_reconstruction_loss + cross_reconstruction_loss) / 2
                    else:
                        cross_reconstruction_loss = 0.
                        recon_loss = self_reconstruction_loss
                else:
                    cross_reconstruction_loss = 0.
                    recon_loss = self_reconstruction_loss

                loss = recon_loss + self.kl * kl_loss
                tloss[0] += float(self_reconstruction_loss)
                tloss[1] += float(kl_loss)
                tloss[2] += float(cross_reconstruction_loss)
                tloss[-1] += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss, ZU, ZV

    @staticmethod
    def generate_sample(baseline_age, age):
        sample = []
        for index, base_a in enumerate(baseline_age):
            match_ba = [i for i, ba in enumerate(baseline_age) if 1e-5 < np.abs(ba - base_a) <= 0.2]
            if match_ba:
                sample.append([index, match_ba])
        result = []
        for index, match in sample:
            match_age = [i for i in match if 1e-5 < np.abs(age[i] - age[index]) <= 0.5]
            for ind in match_age:
                result.append([index, ind])
        index0 = [idx[0] for idx in result]
        index1 = [idx[1] for idx in result]
        return index0, index1

    def generative_parameter_update(self, Z, ZU, ZV):
        X = Variable(self.X).to(self.device).float()
        Y = Variable(self.Y).to(self.device).float()
        Z = Variable(Z).to(self.device).float()
        ZU = Variable(ZU).to(self.device).float()
        ZV = Variable(ZV).to(self.device).float()
        N = X.size()[0]

        xt = torch.transpose(X, 0, 1)
        yt = torch.transpose(Y, 0, 1)
        zt = torch.transpose(Z, 0, 1)
        xtx = torch.matmul(xt, X)
        xtx_inv = torch.inverse(torch.matmul(xt, X))
        yty = torch.matmul(yt, Y)
        ztz = torch.matmul(zt, Z)

        xt_zu = torch.matmul(xt, ZU)
        yt_zv = torch.matmul(yt, ZV)

        for epoch in range(10):
            # updata beta and b
            # H = torch.matmul(torch.matmul(Y, self.D), yt) + self.sigma0_2 * torch.eye(N, device=self.device).float()
            # H_inv = torch.inverse(H)
            # xt_hi_x = torch.matmul(torch.matmul(xt, H_inv), X)
            # xt_hi_z = torch.matmul(torch.matmul(xt, H_inv), Z)
            # mat0 = xt_hi_x + 1 / self.sigma1_2 * xtx
            # mat1 = xt_hi_z + 1 / self.sigma1_2 * xt_zu
            # self.beta = torch.matmul(torch.inverse(mat0), mat1)

            mat0 = self.sigma1_2 * torch.matmul(xt, Z - torch.matmul(Y, self.b))
            self.beta = 1 / (self.sigma0_2 + self.sigma1_2) * torch.matmul(xtx_inv, mat0 + self.sigma0_2 * xt_zu)

            xbeta = torch.matmul(X, self.beta)
            yt_z_xbeta = torch.matmul(yt, Z - xbeta)
            temp_mat = (self.sigma0_2 + self.sigma2_2) * yty - 2 * self.sigma0_2 * self.sigma2_2 * torch.inverse(self.D)
            temp_mat = torch.inverse(temp_mat)
            self.b = torch.matmul(temp_mat, self.sigma2_2 * yt_z_xbeta + self.sigma0_2 * yt_zv)

            # update variance parameter
            xbeta = torch.matmul(X, self.beta)
            yb = torch.matmul(Y, self.b)
            self.sigma0_2 = 1 / (N * dim_z) * torch.pow(torch.norm(Z - xbeta - yb, p='fro'), 2)
            # self.sigma1_2 = 1 / (N * dim_z) * torch.pow(torch.norm(ZU - xbeta, p='fro'), 2)
            self.sigma2_2 = 1 / (N * dim_z) * torch.pow(torch.norm(ZV - yb, p='fro'), 2)
            self.sigma2_2 = min(self.sigma2_2, 0.1)
            self.sigma1_2 = (self.sigma0_2 + self.sigma2_2) * 5
            self.sigma1_2 = max(10, min(15, self.sigma1_2))

            for i in range(5):
                dbbd = torch.matmul(torch.inverse(self.D),
                                    torch.matmul(self.b,
                                                 torch.matmul(torch.transpose(self.b, 0, 1), torch.inverse(self.D))))
                grad_d = -1 / 2 * (dim_z * torch.inverse(self.D) - dbbd)
                self.D = self.D + 1e-5 * grad_d

            # update U and V
            zt_xbeta = torch.matmul(zt, torch.matmul(X, self.beta))
            zt_yb = torch.matmul(zt, torch.matmul(Y, self.b))
            for i in range(5):
                vvt = torch.matmul(self.V, torch.transpose(self.V, 0, 1))
                uut = torch.matmul(self.U, torch.transpose(self.U, 0, 1))
                self.U = torch.matmul(torch.inverse(ztz + self.sigma1_2 * self.lam * vvt), zt_xbeta)
                self.V = torch.matmul(torch.inverse(ztz + self.sigma2_2 * self.lam * uut), zt_yb)

            xt_zu = torch.matmul(xt, torch.matmul(Z, self.U))
            yt_zv = torch.matmul(yt, torch.matmul(Z, self.V))
        print(self.sigma0_2, self.sigma1_2, self.sigma2_2)
        bt_dinv_b = torch.matmul(torch.matmul(torch.transpose(self.b, 0, 1), torch.inverse(self.D)), self.b)
        log_pb = -1 / 2 * (dim_z * torch.log(torch.det(self.D)) + torch.trace(bt_dinv_b))
        utv = torch.matmul(torch.transpose(self.U, 0, 1), self.V)
        utv_norm_2 = torch.pow(torch.norm(utv, p='fro'), 2).cpu().detach().numpy()
        logger.info(f"||U^T * V||^2 = {utv_norm_2:.4}, log p(b) = {log_pb:.3}")

    @staticmethod
    def plot_distribution(Z, title):
        min_z, mean_z, max_z = [], [], []
        fig, axes = plt.subplots(1, dim_z, figsize=(4 * dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(dim_z):
            z = Z[:, i].cpu().detach().numpy()
            axes[i].hist(z, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(z):.4}\nMean: {np.mean(z):.4}\nMax: {np.max(z):.4}")
            min_z.append(np.min(z))
            mean_z.append(np.mean(z))
            max_z.append(np.max(z))
        for axe in axes:
            axe.set_yticks([])
            axe.set_xlim(left=-2, right=2)
        plt.savefig('visualization/{}_distribution.png'.format(title), bbox_inches='tight')
        plt.close()

