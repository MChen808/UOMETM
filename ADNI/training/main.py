import torch
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import logging
import sys
import os
from ADNI.lib.dataset import Data_preprocess_ADNI, Dataset_adni
import argparse
from ADNI.lib import UOMETM_model as model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--dimz', type=int, default=4, help='dimensionality of latent space')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--number', type=int, default=10242, help='number of downsample dimensionality')
parser.add_argument('--lor', type=int, default=0, help='left (0) or right (1) hemisphere')
input_para = parser.parse_args()

# hyperparameter
device = torch.device(f"cuda:{input_para.cuda}")
fold = input_para.fold
epochs = input_para.epochs
dim_z = input_para.dimz
lr = input_para.lr
batch_size = input_para.bs
number = input_para.number
left_right = input_para.lor


if __name__ == '__main__':
    logger.info(f"Device is {device}")
    logger.info(f"##### Fold {fold + 1}/2 #####\n")

    # make directory
    if not os.path.exists('model'):
        os.mkdir('model')
    if not os.path.exists('visualization'):
        os.mkdir('visualization')

    # load data
    data_generator = Data_preprocess_ADNI(number=number)
    demo_train, demo_test = data_generator.generate_demo_train_test(fold)
    thick_train, thick_test, input_dim = data_generator.generate_thick_train_test(fold)
    logger.info(f"Loaded {len(demo_train['age']) + len(demo_test['age'])} scans")

    Dataset = Dataset_adni
    train = Dataset(thick_train['left'], thick_train['right'], demo_train['age'], demo_train['baseline_age'],
                    demo_train['label'], demo_train['subject'], demo_train['timepoint'])
    test = Dataset(thick_test['left'], thick_test['right'], demo_test['age'], demo_test['baseline_age'],
                   demo_test['label'], demo_test['subject'], demo_test['timepoint'])

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False,
                                               num_workers=0, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
                                              num_workers=0, drop_last=False)
    print('Generating data loader finished...')

    autoencoder = model.UOMETM(dim_z, input_dim, left_right)
    autoencoder.device = device
    if hasattr(autoencoder, 'X'):
        X, Y = data_generator.generate_XY(demo_train)
        X, Y = Variable(X).to(device).float(), Variable(Y).to(device).float()
        autoencoder.X, autoencoder.Y = X, Y
    if hasattr(autoencoder, 'batch_size'):
        autoencoder.batch_size = batch_size
    print(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters())} parameters")

    print('Start training...')
    optimizer_fn = optim.Adam
    optimizer = optimizer_fn(autoencoder.parameters(), lr=lr)
    ZU, ZV = autoencoder.train_(train_loader, test_loader, optimizer=optimizer, num_epochs=epochs)

    left_right = 'left' if left_right == 0 else 'right'
    torch.save(autoencoder, 'model/{}_fold_{}_{}'.format(fold, left_right, autoencoder.name))
    logger.info(f'##### Fold {fold + 1}/2 finished #####\n')
    logger.info('Model saved in model/{}_fold_{}_{}'.format(fold, left_right, autoencoder.name))

