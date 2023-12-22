import torch
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import logging
import sys
import os
import argparse
from dataset import Data_preprocess_starmen, Dataset_starmen
from model import UOMETM

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='cuda device, default 0')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--dimz', type=int, default=4, help='dimensionality of latent space')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--bs', type=int, default=128, help='batch size')
input_para = parser.parse_args()

# hyperparameter
device = torch.device(f"cuda:{input_para.cuda}")
fold = input_para.fold
epochs = input_para.epochs
dim_z = input_para.dimz
lr = input_para.lr
batch_size = input_para.bs


if __name__ == '__main__':
    logger.info(f"Device is {device}")
    logger.info(f"##### Fold {fold + 1}/5 #####\n")

    # load train and test data
    data_generator = Data_preprocess_starmen()
    train_data, test_data = data_generator.generate_train_test(fold)
    logger.info(f"Loaded {len(train_data['path']) + len(test_data['path'])} scans")

    train_data.requires_grad = False
    test_data.requires_grad = False
    Dataset = Dataset_starmen
    train = Dataset(train_data['path'], train_data['subject'], train_data['baseline_age'], train_data['age'],
                    train_data['timepoint'], train_data['first_age'], train_data['alpha'])
    test = Dataset(test_data['path'], test_data['subject'], test_data['baseline_age'], test_data['age'],
                   test_data['timepoint'], test_data['first_age'], test_data['alpha'])

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False,
                                               num_workers=0, drop_last=False, pin_memory=True)

    # load model
    autoencoder = UOMETM(dim_z, len(train_data['path']) // 10, 3, 2)
    autoencoder.device = device
    if hasattr(autoencoder, 'X'):
        X, Y = data_generator.generate_XY(train_data)
        X, Y = Variable(X).to(device).float(), Variable(Y).to(device).float()
        autoencoder.X, autoencoder.Y = X, Y
    if hasattr(autoencoder, 'batch_size'):
        autoencoder.batch_size = batch_size
    if hasattr(autoencoder, 'fold'):
        autoencoder.fold = fold
    print(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters())} parameters")

    # train and save model
    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
    autoencoder.train_(train_loader, test=test, optimizer=optimizer, num_epochs=epochs)
    if not os.path.exists('trained_model'):
        os.mkdir('trained_model')
    torch.save(autoencoder, 'trained_model/{}_fold_{}'.format(fold, autoencoder.name))
    logger.info(f"##### Fold {fold + 1}/5 finished #####\n")
    logger.info("Model saved in trained_model/{}_fold_{}".format(fold, autoencoder.name))