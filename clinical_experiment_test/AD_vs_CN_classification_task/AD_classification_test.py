import torch
from torch.autograd import Variable
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

device = torch.device(f"cuda:0")


if __name__ == '__main__':
    # Step 1: load model and classifier
    autoencoder = torch.load('model/UOMETM_ADNI', map_location=device)
    autoencoder.eval()
    classifier = torch.load('model/classifier', map_location=device)

    # Step 2: load exemplary data
    labels = torch.load('data/labels', map_location=device)
    thickness = torch.load('data/thickness', map_location=device)

    # Step 3: generate orthogonal representations
    # ZU represents global trajectory
    # ZV represents individual trajectory
    input_ = Variable(torch.tensor(thickness)).to(device).float()
    _, _, ZU, ZV, _, _ = autoencoder.forward(input_)

    # Step 4: test classifier
    logger.info(f"##### AD vs. CN classification test for UOMETM-ZV #####")
    classifier.test(ZV, labels)
    logger.info(f"##### AD vs. CN classification test for UOMETM-ZU #####")
    classifier.test(ZU, labels)
    logger.info(f"##### AD vs. CN classification test for UOMETM-ZU+V #####")
    classifier.test(ZU + ZV, labels)
