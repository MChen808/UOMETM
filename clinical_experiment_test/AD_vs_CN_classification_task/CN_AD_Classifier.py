import torch.nn as nn
import torch
from torch.autograd import Variable


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        nn.Module.__init__(self)
        self.name = 'Classifier'

        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        return x

    def train_(self, ZV, labels, optimizer, num_epochs):
        self.to(ZV.device)
        criterion = nn.CrossEntropyLoss()
        labels = labels.to(ZV.device).float()
        highest_acc = 0.

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            input_ = Variable(torch.tensor(ZV)).to(ZV.device).float()
            preds = self.forward(input_)
            loss = criterion(preds.float(), labels.long())
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                predicted_labels = torch.argmax(preds, dim=1)
                train_acc = torch.sum(predicted_labels == labels).item() / predicted_labels.size()[-1] * 100
                if train_acc > highest_acc:
                    highest_acc = train_acc
                print(f'ACC / highest: {train_acc:.2f}% / {highest_acc:.2f}%')

    def test(self, ZV, labels):
        self.to(ZV.device)
        input_ = Variable(torch.tensor(ZV)).to(ZV.device).float().clone().detach()
        preds = self.forward(input_)
        predicted_labels = torch.argmax(preds, dim=1)
        train_acc = torch.sum(predicted_labels == labels).item() / predicted_labels.size()[-1] * 100
        for i in range(ZV.shape[0]):
            pred = 'CN' if predicted_labels[i] == 0 else 'AD'
            target = 'CN' if labels[i] == 0 else 'AD'
            print(f"predicted / target: {pred} / {target}")
        print(f'ACC: {train_acc:.2f}%\n')