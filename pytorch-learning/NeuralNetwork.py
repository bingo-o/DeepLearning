import torch
from torch import optim, nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cuda-able", action="store_true", default=False)
args = parser.parse_args()


batch_size = 32
num_epochs = 20
learning_rate = 1e-2

train_dataset = torchvision.datasets.MNIST("./mnist/", train=True, transform=torchvision.transforms.ToTensor(),
                                           download=False)
train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x), inplace=True)
        x = F.relu(self.layer2(x), inplace=True)
        x = self.layer3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() and args.cuda_able else "cpu")

model = NeuralNetwork(28 * 28, 300, 100, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

num_samples = train_dataset.targets.size(0)
num_batches = num_samples // batch_size
for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    for i, (img, label) in enumerate(train_loader):
        img = img.view(img.size(0), -1).to(device)
        label = label.to(device)
        out = model(img)
        num_correct = (torch.argmax(out, dim=1) == label).sum()
        total_correct += num_correct
        loss = criterion(out, label)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch[{} / {}], loss: {:.3f}, acc: {:.3f}".format(epoch + 1, num_epochs, total_loss.item() / num_batches,
                                                             total_correct.item() / num_samples))
