import torch
from torch import optim, nn
import string
from sklearn.model_selection import train_test_split
import nltk
nltk.download("names")
from nltk.corpus import names

num_epochs = 20
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data():
    female_file, male_file = names.fileids()
    female_names = names.words(female_file)
    male_names = names.words(male_file)
    dataset = [(name.lower(), 0) for name in female_names] + [(name.lower(), 1) for name in male_names]
    return dataset

chars = string.ascii_lowercase + " " + "-" + "'"

def name2ids(name):
    ids = [chars.index(char) for char in name]
    return ids

class CharGRU(nn.Module):
    def __init__(self, char_size, embedding_size, hidden_size, target_size):
        super(CharGRU, self).__init__()
        self.embedding = nn.Embedding(char_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, target_size)
        
    def forward(self, x):
        embeds = self.embedding(x)
        out, _ = self.gru(embeds)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class Model:
    def __init__(self, num_epochs=num_epochs, learning_rate=learning_rate):
        self.model = CharGRU(len(chars), 64, 128, 2).to(device)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
    
    def train(self, train_dataset):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):
            total_loss = 0
            for name, sex in train_dataset:
                name = torch.LongTensor([name2ids(name)]).to(device)
                sex = torch.LongTensor([sex]).to(device)
                out = self.model(name)
                loss = criterion(out, sex)
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("epoch[{} / {}], loss: {:.3f}".format(epoch + 1, self.num_epochs, total_loss.item() / len(train_dataset)))
            
    def evaluate(self, test_dataset):
        with torch.no_grad():
            total_correct = 0
            for name, sex in test_dataset:
                name = torch.LongTensor([name2ids(name)])
                out = self.model(name)
                sex_predict = torch.argmax(out, dim=1).item()
                if sex_predict == sex:
                    total_correct += 1
            print("Testing accuracy: {:.3f}".format(total_correct / len(test_dataset)))

    def predict(self, name):
        with torch.no_grad():
            name = torch.LongTensor([name2ids(name.lower())])
            out = self.model(name)
            sex = torch.argmax(out, dim=1).item()
            sex = "female" if sex == 0 else "male"
            return sex

if __name__ == "__main__":
    dataset = get_data()
    train_dataset, test_dataset = train_test_split(dataset)
    model = Model()
    model.train(train_dataset)
    model.evaluate(test_dataset)
    print(model.predict("Bingo"))
    print(model.predict("Jack"))