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


class beta_VAE(nn.Module):
    def __init__(self, input_dim, left_right=0):
        super(beta_VAE, self).__init__()
        nn.Module.__init__(self)
        self.name = 'beta_VAE'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = 5.
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
            nn.Tanh(),
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

    def forward(self, input):
        z = self.encoder(input)
        mu = self.mu(z)
        logVar = self.logVar(z)
        if self.training:
            z = self.reparametrize(mu, logVar)
        else:
            z = mu
        output = self.decoder(z)
        return mu, logVar, output

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2).to(self.device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std=1).to(self.device)
        if self.beta != 0:  # beta VAE
            return mu + eps * std
        else:  # regular AE
            return mu

    @staticmethod
    def loss(mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.mean(-1 - logVar + mu.pow(2) + logVar.exp())
        recon_error = torch.mean((reconstructed - input_) ** 2)
        return recon_error, kl_divergence

    def train_(self, train_data_loader, test_data_loader, optimizer, num_epochs):
        self.to(self.device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('#### Epoch {}/{} ####'.format(epoch + 1, num_epochs))

            tloss = np.array([0., 0., 0.])
            nb_batches = 0

            for data in train_data_loader:
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 labels, 5 subject, 6 timepoint
                image = data[self.left_right]
                optimizer.zero_grad()

                # self-reconstruction loss
                input_ = Variable(image).to(self.device).float()
                mu, logVar, reconstructed = self.forward(input_)
                self_reconstruction_loss, kl_divergence = self.loss(mu, logVar, reconstructed, input_)

                loss = self_reconstruction_loss + self.beta * kl_divergence
                loss.backward()
                optimizer.step()
                tloss[0] += float(self_reconstruction_loss)
                tloss[1] += float(kl_divergence)
                tloss[-1] += float(loss)
                nb_batches += 1

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test_data_loader) if epoch >= num_epochs - 0 else 0.0
            if epoch_loss[-1] <= best_loss:
                es = 0
                best_loss = epoch_loss[-1]
            else:
                es += 1

            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss}/{test_loss} take {end_time - start_time:.3} seconds\n")

    def evaluate(self, test_data_loader):
        self.to(self.device)
        self.training = False
        self.eval()
        tloss = np.array([0., 0., 0.])
        nb_batches = 0

        with torch.no_grad():
            for data in test_data_loader:
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 labels, 5 subject, 6 timepoint
                image = data[self.left_right]

                # self-reconstruction loss
                input_ = Variable(image).to(self.device).float()
                mu, logVar, reconstructed = self.forward(input_)
                self_reconstruction_loss, kl_divergence = self.loss(mu, logVar, reconstructed, input_)

                loss = self_reconstruction_loss + self.beta * kl_divergence

                tloss[0] += float(self_reconstruction_loss)
                tloss[1] += float(kl_divergence)
                tloss[-1] += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss


class ML_VAE(nn.Module):
    def __init__(self, input_dim, left_right=0):
        super(ML_VAE, self).__init__()
        nn.Module.__init__(self)
        self.name = 'ML_VAE'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = 5.
        self.input_dim = input_dim
        self.left_right = left_right

        self.encoder_z = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.mu_style = nn.Sequential(
            nn.Linear(128, dim_z),
            nn.Tanh(),
        )

        self.logVar_style = nn.Sequential(
            nn.Linear(128, dim_z),
            # nn.BatchNorm1d(dim_z),
        )

        self.mu_class = nn.Sequential(
            nn.Linear(128, dim_z),
            nn.Tanh(),
        )

        self.logVar_class = nn.Sequential(
            nn.Linear(128, dim_z),
            # nn.BatchNorm1d(dim_z),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_z * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.input_dim),
            nn.ReLU()
        )

    def encoder(self, image):
        z = self.encoder_z(image)

        # style
        style_mu = self.mu_style(z)
        style_logVar = self.logVar_style(z)

        # class
        class_mu = self.mu_class(z)
        class_logVar = self.logVar_class(z)
        return style_mu, style_logVar, class_mu, class_logVar

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2).to(self.device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std=1).to(self.device)
        return mu + eps * std

    def forward(self, image):
        style_mu, style_logVar, class_mu, class_logVar = self.encoder(image)
        if self.training:
            style_encoded = self.reparametrize(style_mu, style_logVar)
            class_encoded = self.reparametrize(class_mu, class_logVar)
        else:
            style_encoded = style_mu
            class_encoded = class_mu
        return style_mu, style_logVar, class_mu, class_logVar, style_encoded, class_encoded

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.mean(-1 - logVar + mu.pow(2) + logVar.exp())
        recon_error = torch.mean((reconstructed - input_) ** 2)
        return recon_error, kl_divergence

    def train_(self, data_loader, test_data_loader, optimizer, num_epochs):

        self.to(self.device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

            tloss = np.array([0., 0., 0.])
            nb_batches = 0

            ZU, ZV = None, None
            for data in data_loader:
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 labels, 5 subject, 6 timepoint
                image = data[self.left_right]
                optimizer.zero_grad()

                input_ = Variable(image).to(self.device).float()
                style_mu, style_logVar, class_mu, class_logVar, style_encoded, class_encoded = self.forward(input_)
                encoded = torch.cat((style_encoded, class_encoded), dim=1)
                reconstructed = self.decoder(encoded)
                reconstruction_loss, style_kl_loss = self.loss(style_mu, style_logVar, input_, reconstructed)
                reconstruction_loss, class_kl_loss = self.loss(class_mu, class_logVar, input_, reconstructed)
                loss = reconstruction_loss + self.beta * (style_kl_loss + class_kl_loss)

                # store ZU, ZV
                if ZU is None:
                    ZU, ZV = class_encoded, style_encoded
                else:
                    ZU = torch.cat((ZU, class_encoded), 0)
                    ZV = torch.cat((ZV, style_encoded), 0)

                loss.backward()
                optimizer.step()
                tloss[0] += float(reconstruction_loss)
                tloss[1] += float(style_kl_loss + class_kl_loss)
                tloss[-1] += float(loss)
                nb_batches += 1

            # self.plot_distribution(Z, title='Z')
            self.plot_distribution(ZU, title='ZU')
            self.plot_distribution(ZV, title='ZV')

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test_data_loader) if epoch >= num_epochs - 0 else 0.0
            if epoch_loss[-1] <= best_loss:
                es = 0
                best_loss = epoch_loss[-1]
            else:
                es += 1

            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss}/{test_loss} take {end_time - start_time:.3} seconds\n")

        print('Complete training')
        return

    def evaluate(self, test_data_loader):
        self.to(self.device)
        self.training = False
        self.eval()
        tloss = np.array([0., 0., 0.])
        nb_batches = 0

        with torch.no_grad():
            for data in test_data_loader:
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 labels, 5 subject, 6 timepoint
                image = data[self.left_right]

                input_ = Variable(image).to(self.device).float()
                style_mu, style_logVar, class_mu, class_logVar, style_encoded, class_encoded = self.forward(input_)
                encoded = torch.cat((style_encoded, class_encoded), dim=1)
                reconstructed = self.decoder(encoded)
                reconstruction_loss, style_kl_loss = self.loss(style_mu, style_logVar, input_, reconstructed)
                reconstruction_loss, class_kl_loss = self.loss(class_mu, class_logVar, input_, reconstructed)
                loss = reconstruction_loss + self.beta * (style_kl_loss + class_kl_loss)

                tloss[0] += float(reconstruction_loss)
                tloss[1] += float(style_kl_loss + class_kl_loss)
                tloss[-1] += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss

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
            axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/{}_distribution.png'.format(title), bbox_inches='tight')


class rank_VAE(nn.Module):
    def __init__(self, input_dim, left_right=0):
        super(rank_VAE, self).__init__()
        nn.Module.__init__(self)
        self.name = 'rank_VAE'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = 5
        self.gamma = 1
        self.input_dim = input_dim
        self.left_right = left_right

        self.encoder_z = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.mu_zs = nn.Sequential(
            nn.Linear(128, dim_z - 1),
            nn.Tanh(),
        )

        self.logVar_zs = nn.Sequential(
            nn.Linear(128, dim_z - 1),
            # nn.BatchNorm1d(dim_z),
        )

        self.mu_zpsi = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        self.logVar_zpsi = nn.Sequential(
            nn.Linear(128, 1),
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

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2).to(self.device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std=1).to(self.device)
        return mu + eps * std

    def forward(self, image):
        z = self.encoder_z(image)
        zs_mu, zs_logVar = self.mu_zs(z), self.logVar_zs(z)
        zpsi_mu, zpsi_logVar = self.mu_zpsi(z), self.logVar_zpsi(z)
        if self.training:
            zs_encoded = self.reparametrize(zs_mu, zs_logVar)
            zpsi_encoded = self.reparametrize(zpsi_mu, zpsi_logVar)
        else:
            zs_encoded = zs_mu
            zpsi_encoded = zpsi_mu
        return zs_mu, zs_logVar, zpsi_mu, zpsi_logVar, zs_encoded, zpsi_encoded

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.mean(-1 - logVar + mu.pow(2) + logVar.exp())
        recon_error = torch.mean((reconstructed - input_) ** 2)
        return recon_error, kl_divergence

    def rank_loss(self, subject, timepoint, zpsi):
        rank_loss = 0
        num = 0.0

        subject = torch.squeeze(subject)
        timepoint = torch.squeeze(timepoint)
        unique_subject, cnt = torch.unique(subject, return_counts=True)
        select_subject = [s.squeeze() for i, s in enumerate(unique_subject) if cnt[i] >= 2]
        if len(select_subject) == 0:
            return torch.tensor(0.0, device=self.device)
        else:
            select_index = [np.squeeze(np.nonzero(subject == sub)) for sub in select_subject]
            for index in select_index:
                select_tp = timepoint[index].squeeze()
                rank_tp = torch.argsort(select_tp)
                select_zpsi = zpsi[index].squeeze()
                rand_zpsi = torch.argsort(select_zpsi)
                rank_loss += torch.mean((rand_zpsi - rank_tp).float() ** 2)
                num += len(index)
            return rank_loss / num

    def train_(self, data_loader, test_data_loader, optimizer, num_epochs):

        self.to(self.device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

            tloss = np.array([0., 0., 0., 0.])
            nb_batches = 0

            ZU, ZV = None, None
            for data in data_loader:
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 labels, 5 subject, 6 timepoint
                image = data[self.left_right]
                subject = torch.tensor(data[5], device=self.device)
                timepoint = torch.tensor(data[6], device=self.device)
                optimizer.zero_grad()

                input_ = Variable(image).to(self.device).float()
                zs_mu, zs_logVar, zpsi_mu, zpsi_logVar, zs_encoded, zpsi_encoded = self.forward(input_)
                encoded = torch.cat((zs_encoded, zpsi_encoded), dim=1)
                reconstructed = self.decoder(encoded)
                reconstruction_loss, zs_kl_loss = self.loss(zs_mu, zs_logVar, input_, reconstructed)
                reconstruction_loss, zpsi_kl_loss = self.loss(zpsi_mu, zpsi_logVar, input_, reconstructed)
                rank_loss = self.rank_loss(subject, timepoint, zpsi_encoded)

                loss = reconstruction_loss + self.beta * (zs_kl_loss + zpsi_kl_loss) + self.gamma * rank_loss

                # store ZU, ZV
                if ZU is None:
                    ZU, ZV = zpsi_encoded, zs_encoded
                else:
                    ZU = torch.cat((ZU, zpsi_encoded), 0)
                    ZV = torch.cat((ZV, zs_encoded), 0)

                loss.backward()
                optimizer.step()
                tloss[0] += float(reconstruction_loss)
                tloss[1] += float(zs_kl_loss + zpsi_kl_loss)
                tloss[2] += float(rank_loss)
                tloss[-1] += float(loss)
                nb_batches += 1

            # self.plot_distribution(Z, title='Z')
            self.plot_distribution(ZU, title='ZU')
            self.plot_distribution(ZV, title='ZV')

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test_data_loader) if epoch >= num_epochs - 0 else 0.0
            if epoch_loss[-1] <= best_loss:
                es = 0
                best_loss = epoch_loss[-1]
            else:
                es += 1

            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss}/{test_loss} take {end_time - start_time:.3} seconds\n")

        print('Complete training')
        return

    def evaluate(self, test_data_loader):
        self.to(self.device)
        self.training = False
        self.eval()
        tloss = np.array([0., 0., 0., 0.])
        nb_batches = 0

        with torch.no_grad():
            for data in test_data_loader:
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 labels, 5 subject, 6 timepoint
                image = data[self.left_right]
                subject = torch.tensor(data[5], device=self.device)
                timepoint = torch.tensor(data[6], device=self.device)

                input_ = Variable(image).to(self.device).float()
                zs_mu, zs_logVar, zpsi_mu, zpsi_logVar, zs_encoded, zpsi_encoded = self.forward(input_)
                encoded = torch.cat((zs_encoded, zpsi_encoded), dim=1)
                reconstructed = self.decoder(encoded)
                reconstruction_loss, zs_kl_loss = self.loss(zs_mu, zs_logVar, input_, reconstructed)
                reconstruction_loss, zpsi_kl_loss = self.loss(zpsi_mu, zpsi_logVar, input_, reconstructed)
                rank_loss = self.rank_loss(subject, timepoint, zpsi_encoded)

                loss = reconstruction_loss + self.beta * (zs_kl_loss + zpsi_kl_loss) + self.gamma * rank_loss

                tloss[0] += float(reconstruction_loss)
                tloss[1] += float(zs_kl_loss + zpsi_kl_loss)
                tloss[2] += float(rank_loss)
                tloss[-1] += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss

    @staticmethod
    def plot_distribution(Z, title):
        min_z, mean_z, max_z = [], [], []
        fig, axes = plt.subplots(1, dim_z, figsize=(4 * dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(Z.size()[-1]):
            z = Z[:, i].cpu().detach().numpy()
            axes[i].hist(z, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(z):.4}\nMean: {np.mean(z):.4}\nMax: {np.max(z):.4}")
            min_z.append(np.min(z))
            mean_z.append(np.mean(z))
            max_z.append(np.max(z))
        for axe in axes:
            axe.set_yticks([])
            axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/{}_distribution.png'.format(title), bbox_inches='tight')


class LNE(nn.Module):
    def __init__(self, input_dim, left_right=0):
        super(LNE, self).__init__()
        nn.Module.__init__(self)
        self.name = 'LNE'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.lambda_proto = 1.0
        self.lambda_dir = 1.0
        self.input_dim = input_dim
        self.left_right = left_right

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, dim_z),
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

        self.I = 660 if left_right == 0 else 899
        self.N_km = [self.I // 5, self.I // 10, self.I // 20]
        self.num_nb = 5
        self.sample_idx_list = None
        self.concentration_list = None
        self.prototype_list = None

    def forward(self, img1, img2=None):
        bs = img1.shape[0]
        if img2 is None:
            zs = self.encoder(img1)
            recons = self.decoder(zs)
            return zs, recons
        else:
            zs = self.encoder(torch.cat([img1, img2], 0))
            recons = self.decoder(zs)
            zs_flatten = zs.view(bs * 2, -1)
            z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
            recon1, recon2 = recons[:bs], recons[bs:]
            return [z1, z2], [recon1, recon2]

    def build_graph_batch(self, zs):
        z1 = zs[0]
        bs = z1.shape[0]
        dis_mx = torch.zeros(bs, bs).to(self.device)
        for i in range(bs):
            for j in range(i + 1, bs):
                dis_mx[i, j] = torch.sum((z1[i] - z1[j]) ** 2)
                dis_mx[j, i] = dis_mx[i, j]
        sigma = (torch.sort(dis_mx)[0][:, -1]) ** 0.5 - (torch.sort(dis_mx)[0][:, 1]) ** 0.5
        adj_mx = torch.exp(-dis_mx / (2 * sigma ** 2))
        if self.num_nb < bs:
            adj_mx_filter = torch.zeros(bs, bs).to(self.device)
            for i in range(bs):
                ks = torch.argsort(dis_mx[i], descending=False)[:self.num_nb + 1]
                adj_mx_filter[i, ks] = adj_mx[i, ks]
                adj_mx_filter[i, i] = 0.
            return adj_mx_filter
        else:
            return adj_mx * (1. - torch.eye(bs, bs).to(self.device))

    def build_graph_dataset(self, zs_all, zs):
        z1_all = zs_all[0]
        z1 = zs[0]
        ds = z1_all.shape[0]
        bs = z1.shape[0]
        dis_mx = torch.zeros(bs, ds).to(self.device)
        for i in range(bs):
            for j in range(ds):
                dis_mx[i, j] = torch.sum((z1[i] - z1_all[j]) ** 2)
        # sigma = (torch.sort(dis_mx)[0][:, -1])**0.5 - (torch.sort(dis_mx)[0][:, 1])**0.5
        adj_mx = torch.exp(-dis_mx / 100)
        # adj_mx = torch.exp(-dis_mx / (2*sigma**2))
        if self.num_nb < bs:
            adj_mx_filter = torch.zeros(bs, ds).to(self.device)
            for i in range(bs):
                ks = torch.argsort(dis_mx[i], descending=False)[:self.num_neighbours + 1]
                adj_mx_filter[i, ks] = adj_mx[i, ks]
            return adj_mx_filter
        else:
            return adj_mx * (1. - torch.eye(bs, bs).to(self.device))

    @staticmethod
    def compute_social_pooling_delta_z_batch(zs, interval, adj_mx):
        z1, z2 = zs[0], zs[1]
        delta_z = (z2 - z1) / interval.unsqueeze(1)  # [bs, ls]
        delta_h = torch.matmul(adj_mx, delta_z) / adj_mx.sum(1, keepdim=True)  # [bs, ls]
        return delta_z, delta_h

    @staticmethod
    def compute_social_pooling_delta_z_dataset(zs_all, interval_all, zs, interval, adj_mx):
        z1, z2 = zs[0], zs[1]
        delta_z = (z2 - z1) / interval.unsqueeze(1)  # [bs, ls]
        z1_all, z2_all = zs_all[0], zs_all[1]
        delta_z_all = (z2_all - z1_all) / interval_all.unsqueeze(1)  # [bs, ls]
        delta_h = torch.matmul(adj_mx, delta_z_all) / adj_mx.sum(1, keepdim=True)  # [bs, ls]
        return delta_z, delta_h

    def minimatch_sampling_strategy(self, cluster_centers_list, cluster_ids_list):
        # compute distance between clusters
        cluster_dis_ids_list = []

        for m in range(len(cluster_centers_list)):
            cluster_centers = cluster_centers_list[m]
            n_km = cluster_centers.shape[0]
            cluster_dis_ids = np.zeros((n_km, n_km))
            for i in range(n_km):
                dis_cn = np.sqrt(np.sum((cluster_centers[i].reshape(1, -1) - cluster_centers) ** 2, 1))
                cluster_dis_ids[i] = np.argsort(dis_cn)
            cluster_dis_ids_list.append(cluster_dis_ids)

        n_batch = np.ceil(self.I / self.batch_size).astype(int)
        sample_idx_list = []
        for nb in range(n_batch):
            m_idx = np.random.choice(len(cluster_centers_list))  # select round of kmeans
            c_idx = np.random.choice(cluster_centers_list[m_idx].shape[0])  # select a cluster
            sample_idx_batch = []
            n_s_b = 0
            for c_idx_sel in cluster_dis_ids_list[m_idx][
                c_idx]:  # get nbr clusters given distance to selected cluster c_idx
                sample_idx = np.where(cluster_ids_list[m_idx] == c_idx_sel)[0]
                if n_s_b + sample_idx.shape[0] >= self.batch_size:
                    sample_idx_batch.append(np.random.choice(sample_idx, self.batch_size - n_s_b, replace=False))
                    break
                else:
                    sample_idx_batch.append(sample_idx)
                    n_s_b += sample_idx.shape[0]

            sample_idx_batch = np.concatenate(sample_idx_batch, 0)
            sample_idx_list.append(sample_idx_batch)

        sample_idx_list = np.concatenate(sample_idx_list, 0)
        self.sample_idx_list = sample_idx_list[:self.I]

    @staticmethod
    def compute_recon_loss(x, recon):
        return torch.mean((recon - x) ** 2)

    @staticmethod
    def compute_direction_loss(delta_z, delta_h):
        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        delta_h_norm = torch.norm(delta_h, dim=1) + 1e-12
        cos = torch.sum(delta_z * delta_h, 1) / (delta_z_norm * delta_h_norm)
        return (1. - cos).mean()

    def update_kmeans(self, z1_list, cluster_ids_list, cluster_centers_list):
        z1_list = torch.tensor(z1_list).to(self.device)
        self.prototype_list = [torch.tensor(c).to(self.device) for c in cluster_centers_list]
        self.concentration_list = []
        for m in range(len(self.N_km)):  # for each round of kmeans
            prototypes = self.prototype_list[m]
            cluster_ids = cluster_ids_list[m]
            concentration_m = []
            for c in range(self.N_km[m]):  # for each cluster center
                zs = z1_list[cluster_ids == c]
                n_c = zs.shape[0]
                norm = torch.norm(zs - prototypes[c].view(1, -1), dim=1).sum()
                concentration = norm / (n_c * math.log(n_c + 10))
                concentration_m.append(concentration)
            self.concentration_list.append(torch.tensor(concentration_m).to(self.device))

    def compute_prototype_NCE(self, z1, cluster_ids):
        loss = 0
        for m in range(len(self.N_km)):  # for each round of kmeans
            prototypes_sel = self.prototype_list[m][cluster_ids[m]]
            concentration_sel = self.concentration_list[m][cluster_ids[m]]
            nominator = torch.sum(z1 * prototypes_sel / concentration_sel.view(-1, 1), 1)
            denominator = torch.logsumexp(torch.matmul(z1, torch.transpose(self.prototype_list[m], 0, 1)) /
                                          self.concentration_list[m].view(1, self.N_km[m]), dim=1)
            loss += -(nominator - denominator).mean()
        return loss / (len(self.N_km) * z1.shape[0])

    def train_(self, data_loader, test_data_loader, optimizer, num_epochs):
        self.to(self.device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('#### Epoch {}/{} ####'.format(epoch + 1, num_epochs))

            tloss = np.array([0., 0., 0., 0.])
            nb_batches = 0

            # k-means for z1
            with torch.no_grad():
                self.eval()
                z1_list = []
                for data in data_loader:
                    image = torch.tensor(data[self.left_right]).to(self.device).float()
                    z1 = self.encoder(image)
                    z1_list.append(z1.view(image.shape[0], -1))
                z1_list = torch.cat(z1_list).detach().cpu().numpy()
                print('Finished computing z1 for all training samples!')

                cluster_ids_list = []
                cluster_centers_list = []
                for n_km in self.N_km:
                    kmeans = KMeans(n_clusters=n_km, n_init="auto").fit(z1_list)
                    cluster_centers = kmeans.cluster_centers_
                    cluster_ids = kmeans.labels_
                    cluster_ids_list.append(cluster_ids)
                    cluster_centers_list.append(cluster_centers)
                print('Finished K-means clustering')

            self.update_kmeans(z1_list, cluster_ids_list, cluster_centers_list)
            self.minimatch_sampling_strategy(cluster_centers_list, cluster_ids_list)

            # training
            for iter, data in tqdm(enumerate(data_loader)):
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 labels, 5 subject, 6 timepoint
                image = Variable(data[self.left_right]).to(self.device).float()
                age = torch.tensor(data[2]).to(self.device).float().squeeze()
                optimizer.zero_grad()

                bs = image.size()[0]
                idx1 = torch.arange(0, bs - 1)
                idx2 = idx1 + 1

                img1 = image[idx1]
                img2 = image[idx2]
                interval = age[idx2] - age[idx1]
                cluster_ids = [cluster_ids_list[m][iter * self.batch_size:(iter + 1) * self.batch_size] for m in
                               range(len(self.N_km))]
                cluster_ids = [c[:-1] for c in cluster_ids]

                zs, recons = self.forward(img1, img2)
                adj_mx = self.build_graph_batch(zs)
                delta_z, delta_h = self.compute_social_pooling_delta_z_batch(zs, interval, adj_mx)

                print(data[5], data[6], age)

                loss_recon = 0.5 * (self.compute_recon_loss(img1, recons[0]) + self.compute_recon_loss(img2, recons[1]))
                loss_dir = self.compute_direction_loss(delta_z, delta_h)
                loss_proto = self.compute_prototype_NCE(zs[0], cluster_ids)

                loss = loss_recon + self.lambda_dir * loss_dir + self.lambda_proto * loss_proto
                loss.backward()
                print(loss_recon.item(), loss_dir, loss_proto)

                optimizer.step()
                tloss[0] += float(loss_recon)
                tloss[1] += float(loss_dir)
                tloss[2] += float(loss_proto)
                tloss[-1] += float(loss)
                nb_batches += 1

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test_data_loader) if epoch >= num_epochs - 0 else 0.0
            if epoch_loss[-1] <= best_loss:
                es = 0
                best_loss = epoch_loss[-1]
            else:
                es += 1

            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss}/{test_loss} take {end_time - start_time:.3} seconds\n")

    def evaluate(self, test_data_loader):
        self.to(self.device)
        self.training = False
        self.eval()
        tloss = np.array([0., 0., 0., 0.])
        nb_batches = 0

        with torch.no_grad():
            z1_list = []
            for data in test_data_loader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                z1 = self.encoder(image)
                z1_list.append(z1.view(image.shape[0], -1))
            z1_list = torch.cat(z1_list).detach().cpu().numpy()

            cluster_ids_list = []
            cluster_centers_list = []
            for n_km in self.N_km:
                kmeans = KMeans(n_clusters=n_km, n_init="auto").fit(z1_list)
                cluster_centers = kmeans.cluster_centers_
                cluster_ids = kmeans.labels_
                cluster_ids_list.append(cluster_ids)
                cluster_centers_list.append(cluster_centers)

            self.update_kmeans(z1_list, cluster_ids_list, cluster_centers_list)
            self.minimatch_sampling_strategy(cluster_centers_list, cluster_ids_list)

            for iter, data in tqdm(enumerate(test_data_loader)):
                # data: 0 lthick, 1 rthick, 2 age, 3 baseline_age, 4 labels, 5 subject, 6 timepoint
                image = torch.tensor(data[self.left_right]).to(self.device).float()
                age = torch.tensor(data[2]).to(self.device).float().squeeze()

                bs = image.size()[0]
                idx1 = torch.arange(0, bs - 1)
                idx2 = idx1 + 1

                img1 = image[idx1]
                img2 = image[idx2]
                interval = age[idx2] - age[idx1]
                cluster_ids = [cluster_ids_list[m][iter * self.batch_size:(iter + 1) * self.batch_size] for m in
                               range(len(self.N_km))]
                cluster_ids = [c[:-1] for c in cluster_ids]

                zs, recons = self.forward(img1, img2)
                adj_mx = self.build_graph_batch(zs)
                delta_z, delta_h = self.compute_social_pooling_delta_z_batch(zs, interval, adj_mx)

                loss_recon = 0.5 * (self.compute_recon_loss(img1, recons[0]) + self.compute_recon_loss(img2, recons[1]))
                loss_dir = self.compute_direction_loss(delta_z, delta_h)
                loss_proto = self.compute_prototype_NCE(zs[0], cluster_ids)

                loss = loss_recon + self.lambda_dir * loss_dir + self.lambda_proto * loss_proto

                tloss[0] += float(loss_recon)
                tloss[1] += float(loss_dir)
                tloss[2] += float(loss_proto)
                tloss[-1] += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss