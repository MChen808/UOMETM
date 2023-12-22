import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
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


class UOMETM(nn.Module):
    def __init__(self, dim_z, M, p, q):
        # self.dim_z: dimensionality of the latent space
        # M: number of subjects
        # p, q: number of fixed and ramdom effects variables

        super(UOMETM, self).__init__()
        nn.Module.__init__(self)
        self.name = 'UOMETM'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = 1
        self.lam = 1.0
        self.dim_z = dim_z

        # encoder networks
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 16 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 32 x 8 x 8
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc10 = nn.Linear(2048, self.dim_z)

        # decoder networks
        self.fc3 = nn.Linear(self.dim_z, 512)
        self.upconv1 = nn.ConvTranspose2d(8, 64, 3, stride=2, padding=1, output_padding=1)  # 64 x 16 x 16
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 32 x 32 x 32
        self.upconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 1 x 64 x 64
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

        # save losses
        self.train_loss, self.test_loss = [], []
        self.self_recon, self.cross_recon, self.modeling, self.log_p_b = [], [], [], []

        # matrices to calculate the modeling loss
        self.X, self.Y = None, None
        self.beta = torch.rand(size=[p, self.dim_z], device=self.device)
        self.b = torch.normal(mean=0, std=1, size=[q * M, self.dim_z], device=self.device)
        self.U = torch.diag(torch.tensor([1 for i in range(self.dim_z // 2)] + [0 for i in range(self.dim_z - self.dim_z // 2)],
                                         device=self.device)).float()
        self.V = torch.eye(self.dim_z, device=self.device) - self.U
        self.sigma0_2, self.sigma1_2, self.sigma2_2 = 1, 1, 1
        self.D = torch.eye(q * M, device=self.device).float()

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        z = torch.tanh(self.fc10(h3.view(h3.size()[0], -1)))
        return z

    def decoder(self, encoded):
        h6 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 8, 8, 8])
        h7 = F.relu(self.bn4(self.upconv1(h6)))
        h8 = F.relu(self.bn5(self.upconv2(h7)))
        reconstructed = F.relu(self.upconv3(h8))
        return reconstructed

    def forward(self, image0, image1=None):
        z0 = self.encoder(image0)
        zu0 = torch.matmul(z0, self.U)
        zv0 = torch.matmul(z0, self.V)
        if image1 is not None:  # cross-recon loss
            z1 = self.encoder(image1)
            encoded = torch.matmul(z1, self.U) + zv0
            reconstructed = self.decoder(encoded)
            return reconstructed
        else:  # self-recon loss
            encoded = zu0 + zv0
            reconstructed = self.decoder(encoded)
            return reconstructed, z0, zu0, zv0

    @staticmethod
    def loss(input_, reconstructed):
        recon_loss = torch.sum((reconstructed - input_) ** 2) / input_.shape[0]
        return recon_loss

    def train_(self, data_loader, test, optimizer, num_epochs):
        self.to(self.device)
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('#### Epoch {}/{} ####'.format(epoch + 1, num_epochs))

            tloss = 0.0
            nb_batches = 0

            s_tp, Z, ZU, ZV = None, None, None, None
            for data in data_loader:
                image = torch.tensor([[np.load(path)] for path in data[0]], device=self.device).float()
                optimizer.zero_grad()

                # self-reconstruction loss
                input_ = Variable(image).to(self.device)
                reconstructed, z, zu, zv = self.forward(input_)
                self_reconstruction_loss = self.loss(input_, reconstructed)

                # store Z, ZU, ZV
                subject = torch.tensor([[s for s in data[1]]], device=self.device)
                tp = torch.tensor([[tp for tp in data[4]]], device=self.device)
                st = torch.transpose(torch.cat((subject, tp), 0), 0, 1)
                if s_tp is None:
                    s_tp, Z, ZU, ZV = st, z, zu, zv
                else:
                    s_tp = torch.cat((s_tp, st), 0)
                    Z = torch.cat((Z, z), 0)
                    ZU = torch.cat((ZU, zu), 0)
                    ZV = torch.cat((ZV, zv), 0)

                # cross-reconstruction loss
                age = data[3]
                index0, index1 = self.generate_sample(age)
                image0 = image[index0]
                image1 = image[index1]
                if index0:
                    input0_ = Variable(image0).to(self.device)
                    input1_ = Variable(image1).to(self.device)
                    reconstructed = self.forward(input0_, input1_)
                    cross_reconstruction_loss = self.loss(input0_, reconstructed)

                    self.self_recon.append(self_reconstruction_loss.cpu().detach().numpy())
                    self.cross_recon.append(cross_reconstruction_loss.cpu().detach().numpy())
                    recon_loss = (self_reconstruction_loss + cross_reconstruction_loss) / 2
                else:
                    recon_loss = self_reconstruction_loss

                loss = recon_loss
                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1

            # comply with generative model
            sort_index1 = s_tp[:, 1].sort()[1]
            sorted_s_tp = s_tp[sort_index1]
            sort_index2 = sorted_s_tp[:, 0].sort()[1]
            Z, ZU, ZV = Z[sort_index1], ZU[sort_index1], ZV[sort_index1]
            Z, ZU, ZV = Z[sort_index2], ZU[sort_index2], ZV[sort_index2]
            min_, mean_, max_ = self.plot_z_distribution(Z, ZU, ZV)
            self.mixed_effects_modeling(Z, ZU, ZV)

            epoch_loss = tloss / nb_batches
            test_loss = self.evaluate(test)
            self.train_loss.append(epoch_loss)
            self.test_loss.append(test_loss)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1

            # plot result
            if epoch % 5 == 0:
                self.plot_recon(test)
                self.plot_simu_repre(min_, mean_, max_)
                self.plot_grad_simu_repre(min_, mean_, max_)
                self.plot_loss()

            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss:.4}/{test_loss:.4} take {end_time - start_time:.3} seconds\n")

        print('Complete training')
        return

    def evaluate(self, test):
        self.to(self.device)
        self.training = False
        self.eval()
        dataloader = torch.utils.data.DataLoader(test, batch_size=32, num_workers=0, shuffle=False)
        tloss = 0.0
        nb_batches = 0

        with torch.no_grad():
            for data in dataloader:
                image = torch.tensor([[np.load(path)] for path in data[0]]).float()

                # self-reconstruction loss
                input_ = Variable(image).to(self.device)
                reconstructed, z, zu, zv = self.forward(input_)
                self_reconstruction_loss = self.loss(input_, reconstructed)

                # cross-reconstruction loss
                age = data[3]
                index0, index1 = self.generate_sample(age)
                image0 = image[index0]
                image1 = image[index1]
                if index0:
                    input0_ = Variable(image0).to(self.device)
                    input1_ = Variable(image1).to(self.device)
                    reconstructed = self.forward(input0_, input1_)
                    cross_reconstruction_loss = self.loss(input0_, reconstructed)
                    recon_loss = (self_reconstruction_loss + cross_reconstruction_loss) / 2
                else:
                    recon_loss = self_reconstruction_loss

                loss = recon_loss
                tloss += float(loss)
                nb_batches += 1

        loss = tloss / nb_batches
        self.training = True
        return loss

    @staticmethod
    def generate_sample(age):
        pair = []
        for index, a in enumerate(age):
            match_a = [i for i, aa in enumerate(age) if 1e-5 < np.abs(aa - a) <= 0.05]
            if match_a:
                for match in match_a:
                    pair.append([index, match])
        index0 = [idx[0] for idx in pair]
        index1 = [idx[1] for idx in pair]
        return index0, index1

    def plot_recon(self, data, n_subject=3):
        # Plot the reconstruction
        fig, axes = plt.subplots(2 * n_subject, 10, figsize=(20, 4 * n_subject))
        plt.subplots_adjust(wspace=0, hspace=0)
        for j in range(n_subject):
            for i in range(10):
                test_image = torch.tensor(np.load(data[j * 10 + i][0])).resize(1, 1, 64, 64).float()
                test_image = Variable(test_image).to(self.device)
                out, _, _, _ = self.forward(test_image)
                axes[2 * j][i].matshow(255 * test_image[0][0].cpu().detach().numpy())
                axes[2 * j + 1][i].matshow(255 * out[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        if not os.path.exists('visualization/training_process'):
            os.mkdir('visualization/training_process')
        plt.savefig('visualization/training_process/reconstruction.png', bbox_inches='tight')
        plt.close()

    def plot_simu_repre(self, min_, mean_, max_):
        # Plot simulated data in all directions of the latent space
        # Z
        fig, axes = plt.subplots(self.dim_z, 11, figsize=(22, 2 * self.dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(self.dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.U) + torch.matmul(simulated_latent, self.V)
                simulated_img = self.decoder(encoded)
                axes[i][idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        if not os.path.exists('visualization/visualize_latent_space'):
            os.mkdir('visualization/visualize_latent_space')
        plt.savefig('visualization/visualize_latent_space/simulation_Z.png', bbox_inches='tight')
        plt.close()

        # ZU
        fig, axes = plt.subplots(self.dim_z, 11, figsize=(22, 2 * self.dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(self.dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.U)
                simulated_img = self.decoder(encoded)
                axes[i][idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig('visualization/visualize_latent_space/simulation_ZU.png', bbox_inches='tight')
        plt.close()

        # ZV
        fig, axes = plt.subplots(self.dim_z, 11, figsize=(22, 2 * self.dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(self.dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.V)
                simulated_img = self.decoder(encoded)
                axes[i][idx].matshow(255 * simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig('visualization/visualize_latent_space/simulation_ZV.png', bbox_inches='tight')
        plt.close()
        self.training = True

    def plot_grad_simu_repre(self, min_, mean_, max_):
        # Plot the gradient map of simulated data in all directions of the latent space
        # Z
        fig, axes = plt.subplots(self.dim_z, 10, figsize=(20, 2 * self.dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(self.dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.U) + torch.matmul(simulated_latent, self.V)
                simulated_img = self.decoder(encoded)
                if idx == 0:
                    template = simulated_img
                    continue
                grad_img = simulated_img - template
                template = simulated_img
                axes[i][idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        if not os.path.exists('visualization/visualize_latent_space'):
            os.mkdir('visualization/visualize_latent_space')
        plt.savefig('visualization/visualize_latent_space/subtraction_simulation_Z.png', bbox_inches='tight')
        plt.close()

        # ZU
        fig, axes = plt.subplots(self.dim_z, 10, figsize=(20, 2 * self.dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(self.dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.U)
                simulated_img = self.decoder(encoded)
                if idx == 0:
                    template = simulated_img
                    continue
                grad_img = simulated_img - template
                template = simulated_img
                axes[i][idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        plt.savefig('visualization/visualize_latent_space/subtraction_simulation_ZU.png', bbox_inches='tight')
        plt.close()

        # ZV
        fig, axes = plt.subplots(self.dim_z, 10, figsize=(20, 2 * self.dim_z))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(self.dim_z):
            arange = np.linspace(min_[0][i], max_[0][i], num=11)
            for idx, j in enumerate(arange):
                simulated_latent = torch.tensor([[mean for mean in mean_[0]]], device=self.device)
                simulated_latent[0][i] = j
                encoded = torch.matmul(simulated_latent, self.V)
                simulated_img = self.decoder(encoded)
                if idx == 0:
                    template = simulated_img
                    continue
                grad_img = simulated_img - template
                template = simulated_img
                axes[i][idx - 1].matshow(grad_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'),
                                         norm=matplotlib.colors.CenteredNorm())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm()),
            cax=fig.add_axes([0.92, 0.15, 0.01, 0.7]))
        plt.savefig('visualization/visualize_latent_space/subtraction_simulation_ZV.png', bbox_inches='tight')
        plt.close()
        self.training = True

    def plot_z_distribution(self, Z, ZU, ZV):
        min_z, mean_z, max_z = [], [], []
        fig, axes = plt.subplots(1, self.dim_z, figsize=(4 * self.dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(self.dim_z):
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
        if not os.path.exists('visualization/distribution_latent_space'):
            os.mkdir('visualization/distribution_latent_space')
        plt.savefig('visualization/distribution_latent_space/Z_distribution.png', bbox_inches='tight')
        plt.close()

        min_zu, mean_zu, max_zu = [], [], []
        fig, axes = plt.subplots(1, self.dim_z, figsize=(4 * self.dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(self.dim_z):
            zu = ZU[:, i].cpu().detach().numpy()
            axes[i].hist(zu, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(zu):.4}\nMean: {np.mean(zu):.4}\nMax: {np.max(zu):.4}")
            min_zu.append(np.min(zu))
            mean_zu.append(np.mean(zu))
            max_zu.append(np.max(zu))
        for axe in axes:
            axe.set_yticks([])
            axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/distribution_latent_space/ZU_distribution.png', bbox_inches='tight')
        plt.close()

        min_zv, mean_zv, max_zv = [], [], []
        fig, axes = plt.subplots(1, self.dim_z, figsize=(4 * self.dim_z, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for i in range(self.dim_z):
            zv = ZV[:, i].cpu().detach().numpy()
            axes[i].hist(zv, bins=70, density=True)
            axes[i].set_title('{}-th dim'.format(i + 1))
            axes[i].set_xlabel(f"Min: {np.min(zv):.4}\nMean: {np.mean(zv):.4}\nMax: {np.max(zv):.4}")
            min_zv.append(np.min(zv))
            mean_zv.append(np.mean(zv))
            max_zv.append(np.max(zv))
        for axe in axes:
            axe.set_yticks([])
            axe.set_xlim(left=-1, right=1)
        plt.savefig('visualization/distribution_latent_space/ZV_distribution.png', bbox_inches='tight')
        plt.close()

        min_ = [min_z, min_zu, min_zv]
        mean_ = [mean_z, mean_zu, mean_zv]
        max_ = [max_z, max_zu, max_zv]
        return min_, mean_, max_

    def plot_loss(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        plt.subplots_adjust(wspace=0.2, hspace=0.1)

        axes[0][0].plot(self.train_loss, 'red')
        axes[0][0].plot(self.test_loss, 'blue')
        axes[0][0].set_title('Training and Test Loss')
        axes[0][0].legend(['train', 'test'])

        axes[0][1].plot(self.cross_recon, 'red')
        axes[0][1].plot(self.self_recon, 'blue')
        axes[0][1].set_title('Self-Recon & Cross-Recon Loss')
        axes[0][1].legend(['cross-recon', 'self-recon'])
        axes[0][1].set_ylim(bottom=0)

        axes[1][0].plot(self.modeling, 'darkviolet')
        axes[1][0].set_title('Modeling Loss')
        axes[1][0].set_ylim(bottom=0)

        axes[1][1].plot(self.log_p_b)
        axes[1][1].set_title('log p(b)')

        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                # ax.set_yticks([])
                ax.set_xlim(left=0)
        if not os.path.exists('visualization/training_process'):
            os.mkdir('visualization/training_process')
        plt.savefig('visualization/training_process/loss.png', bbox_inches='tight')
        plt.close()

    def mixed_effects_modeling(self, Z, ZU, ZV):
        X = Variable(self.X).to(self.device).float()
        Y = Variable(self.Y).to(self.device).float()
        Z = Variable(Z).to(self.device).float()
        ZU = Variable(ZU).to(self.device).float()
        ZV = Variable(ZV).to(self.device).float()

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
            H = torch.matmul(torch.matmul(Y, self.D), yt) + \
                self.sigma0_2 * torch.eye(X.size()[0], device=self.device).float()
            H_inv = torch.inverse(H)
            xt_hi_x = torch.matmul(torch.matmul(xt, H_inv), X)
            xt_hi_z = torch.matmul(torch.matmul(xt, H_inv), Z)
            mat0 = xt_hi_x + 1 / self.sigma1_2 * xtx
            mat1 = xt_hi_z + 1 / self.sigma1_2 * xt_zu
            self.beta = torch.matmul(torch.inverse(mat0), mat1)

            xbeta = torch.matmul(X, self.beta)
            yt_z_xbeta = torch.matmul(yt, Z - xbeta)
            self.b = torch.matmul(
                torch.inverse(
                    (self.sigma0_2 + self.sigma2_2) * yty - 2 * self.sigma0_2 * self.sigma2_2 * torch.inverse(self.D)),
                self.sigma2_2 * yt_z_xbeta + self.sigma0_2 * yt_zv
            )

            # update variance parameter
            xbeta = torch.matmul(X, self.beta)
            yb = torch.matmul(Y, self.b)
            self.sigma0_2 = 1 / (X.size()[0] * self.dim_z) * torch.pow(torch.norm(Z - xbeta - yb, p='fro'), 2)
            # self.sigma1_2 = 1 / (X.size()[0] * self.dim_z) * torch.pow(torch.norm(ZU - xbeta, p='fro'), 2)
            self.sigma2_2 = 1 / (X.size()[0] * self.dim_z) * torch.pow(torch.norm(ZV - yb, p='fro'), 2)

            for i in range(3):
                dbbd = torch.matmul(torch.inverse(self.D),
                                    torch.matmul(self.b,
                                                 torch.matmul(torch.transpose(self.b, 0, 1), torch.inverse(self.D))))
                grad_d = -1 / 2 * (self.dim_z * torch.inverse(self.D) - dbbd)
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

        bt_dinv_b = torch.matmul(torch.matmul(torch.transpose(self.b, 0, 1), torch.inverse(self.D)), self.b)
        log_pb = -1 / 2 * (self.dim_z * torch.log(torch.det(self.D)) + torch.trace(bt_dinv_b))
        log_pb = log_pb.cpu().detach().numpy()
        if log_pb != np.inf:
            self.log_p_b.append(log_pb)
        utv = torch.matmul(torch.transpose(self.U, 0, 1), self.V)
        utv_norm_2 = torch.pow(torch.norm(utv, p='fro'), 2).cpu().detach().numpy()
        logger.info(f"||U^T * V||^2 = {utv_norm_2:.4}")

        modeling_loss = torch.pow(torch.norm(Z - xbeta - yb, p='fro'), 2) + \
                        torch.pow(torch.norm(ZU - xbeta, p='fro'), 2) + \
                        torch.pow(torch.norm(ZV - yb, p='fro'), 2)
        if modeling_loss < 100:
            self.modeling.append(float(modeling_loss))