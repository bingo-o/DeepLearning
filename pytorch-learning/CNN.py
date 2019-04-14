import torch
from torch import optim, nn
import torch.utils.data as Data
import torchvision

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cuda-able", action="store_true", default=False)
args = parser.parse_args()

batch_size = 128
num_epochs = 20
learning_rate = 1e-2

train_dataset = torchvision.datasets.MNIST("./mnist/", train=True, transform=torchvision.transforms.ToTensor(), download=False)
train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class CNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, n_class)
        )
        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda_able else "cpu")

model = CNN(1, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

num_samples = train_dataset.targets.size(0)
num_batches = num_samples // batch_size
for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    for i, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        num_correct = (torch.argmax(out, dim=1) == label).sum()
        total_correct += num_correct
        loss = criterion(out, label)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch[{} / {}], loss: {:.3f}, acc: {:.3f}".format(epoch + 1,  num_epochs, total_loss.item() / num_batches, total_correct.item() / num_samples))
