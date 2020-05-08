import argparse
import csv

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from radialbnn.radial_layer import RadialLayer
from radialbnn.utils.variational_approximator import variational_approximator


@variational_approximator
class RadialBayesianNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()

        self.rl1 = RadialLayer(input_dim, 200)
        self.rl2 = RadialLayer(200, 200)
        self.rl3 = RadialLayer(200, output_dim)

    def forward(self, x):

        x = x.view(-1, 28 * 28)

        x = F.relu(self.rl1(x))
        x = F.relu(self.rl2(x))
        x = self.rl3(x)

        return x


def train(model, trainloader, optimiser, epoch, device):

    train_loss = 0.0

    model.train()
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)

        optimiser.zero_grad()

        outputs = model(data)
        loss = model.elbo(data, labels, criterion, n_samples=5)
        train_loss += loss.item() * data.size(0)

        loss.backward()
        optimiser.step()

    train_loss /= len(trainloader.dataset)
    return train_loss


def test(model, testloader, device):

    correct = 0
    test_loss = 0.0

    model.eval()
    for batch_idx, (data, labels) in enumerate(testloader):
        data, labels = data.to(device), labels.to(device)

        outputs = model(data)
        loss = F.cross_entropy(outputs, labels)
        test_loss += loss.item() * data.size(0)

        preds = outputs.argmax(dim=1, keepdim=True)
        correct += preds.eq(labels.view_as(preds)).sum().item()

    test_loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)

    return test_loss, accuracy


if __name__ == '__main__':

    # read command line arguments
    parser = argparse.ArgumentParser(
        description='Radial BNN - MNIST Classification Example.'
    )

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--testbatch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: false)')

    args = parser.parse_args()

    # set training device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load / process data
    trainset = datasets.MNIST('./data',
                              train=True,
                              download=True,
                              transform=transform)

    testset = datasets.MNIST('./data',
                             train=False,
                             download=True,
                             transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              **kwargs)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.testbatch_size,
                                             **kwargs)

    model = RadialBayesianNetwork(28 * 28, 10).to(device)
    optimiser = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # prepare results file
    rows = ['epoch', 'train_loss', 'test_loss', 'accuracy']
    with open('results.csv', 'w+', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(rows)

    # run training
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, trainloader, optimiser, epoch, device)
        test_loss, accuracy = test(model, testloader, device)

        _results = [epoch, train_loss, test_loss, accuracy]
        with open('results.csv', 'a', newline="") as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(_results)
