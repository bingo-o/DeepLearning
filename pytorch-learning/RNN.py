import torch
from torch import optim, nn
import torch.utils.data as Data
import torchvision

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cuda-able", action="store_true", default=False)
args = parser.parse_args()

batch_size = 100
num_epochs = 20
learning_rate = 1e-3

train_dataset = torchvision.datasets.MNIST("./mnist/", train=True, transform=torchvision.transforms.ToTensor(), download=False)
train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda_able else "cpu")

model = RNN(28, 128, 2, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_samples = train_dataset.targets.size(0)
num_batches = num_samples // batch_size
for epoch in range(num_epochs):
    total_correct = 0
    total_loss = 0
    for i, (img, label) in enumerate(train_loader):
        img = img.squeeze(1).to(device)
        label = label.to(device)
        out = model(img)
        num_correct = (torch.argmax(out, dim=1) == label).sum()
        total_correct += num_correct
        loss = criterion(out, label)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch[{} / {}], loss: {:.3f}, acc: {:.3f}".format(epoch + 1, num_epochs, total_loss.item() / num_batches, total_correct.item() / num_samples))
