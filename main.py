import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class CNNMnist(nn.Module):

    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_train_test_dataset():
    data_dir = '../data/'
    apply_transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(data_dir,
                                   train=True,
                                   download=True,
                                   transform=apply_transform)
    test_dataset = datasets.MNIST(data_dir,
                                  train=False,
                                  download=True,
                                  transform=apply_transform)

    return train_dataset, test_dataset


def discrepancy(weightsA, weightsB):
    keys = weightsA.keys()
    S_t = len(keys)
    d = 0
    for key in keys:
        w_a = weightsA[key]
        w_b = weightsB[key]
        norm = torch.norm(torch.sub(w_a, w_b))
        d += norm
    return d / S_t


def dataset_to_nodes_partitioning(nodes_count: int, areas: int, random_seed: int, shuffling: bool = False):
    np.random.seed(random_seed)  # set seed from Alchemist to make the partitioning deterministic
    dataset_download_path = "../../build/dataset"
    apply_transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(dataset_download_path, train=True, download=True, transform=apply_transform)

    nodes_per_area = int(nodes_count / areas)
    dataset_labels_count = len(train_dataset.classes)
    split_nodes_per_area = np.array_split(np.arange(nodes_count), areas)
    split_classes_per_area = np.array_split(np.arange(dataset_labels_count), areas)
    nodes_and_classes = zip(split_nodes_per_area, split_classes_per_area)

    index_mapping = {}

    for index, (nodes, classes) in enumerate(nodes_and_classes):
        records_per_class = [index for index, (_, lab) in enumerate(train_dataset) if lab in classes]
        # intra-class shuffling
        if shuffling:
            np.random.shuffle(records_per_class)
        split_record_per_node = np.array_split(records_per_class, nodes_per_area)
        for node in nodes:
            index_mapping[node] = split_record_per_node[node % nodes_per_area].tolist()

    return index_mapping


def train_n_epochs(model, train_data_loader, epochs):
    training_loss = 0
    batch_loss = []
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    for _ in range(epochs):
        for batch_index, (images, labels) in enumerate(train_data_loader):
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            batch_loss.append(loss.item())

    return model.state_dict()

def eval(model, eval_data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in eval_data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def eval_loss(model, eval_data_loader):
    model.eval()
    criterion = nn.NLLLoss()
    loss = 0
    with torch.no_grad():
        for images, labels in eval_data_loader:
            outputs = model(images)
            loss += criterion(outputs, labels).item()
    return loss
def plot_data_discrepancy(dataframe):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epochs'], df['Discrepancy0_1'], label='Discrepancy0_1', marker='o')
    plt.plot(df['Epochs'], df['Discrepancy0_8'], label='Discrepancy0_8', marker='s')
    plt.plot(df['Epochs'], df['Discrepancy1_8'], label='Discrepancy1_8', marker='^')
    plt.xlabel('Epochs')
    plt.ylabel('Discrepancy')
    plt.grid(True)
    plt.legend()
    plt.savefig('discrepancy.png', dpi=500)

def plot_data_eval(dataframe):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epochs'], df['Eval0_1'], label='Eval0_1', marker='o')
    plt.plot(df['Epochs'], df['Eval0_8'], label='Eval0_8', marker='s')
    plt.plot(df['Epochs'], df['Eval1_8'], label='Eval1_8', marker='^')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('accuracy.png', dpi=500)
if __name__ == '__main__':
    train_dataset, test_dataset = get_train_test_dataset()
    mapping = dataset_to_nodes_partitioning(10, 2, 42)

    dataset0 = DataLoader(DatasetSplit(train_dataset, mapping[0]), batch_size=25, shuffle=True)
    dataset1 = DataLoader(DatasetSplit(train_dataset, mapping[1]), batch_size=25, shuffle=True)
    dataset8 = DataLoader(DatasetSplit(train_dataset, mapping[8]), batch_size=25, shuffle=True)

    df = pd.DataFrame(columns=['Epochs', 'Discrepancy0_1', 'Discrepancy0_8', 'Discrepancy1_8'])

def initialize_model():
    model = CNNMnist()
    model.load_state_dict(torch.load('initialmodel'))
    return model

def total_eval_loss(model1, model2, dataset1, dataset2):
    return eval_loss(model1, dataset2) + eval_loss(model2, dataset1)

def calculate_discrepancy(model1, model2):
    return discrepancy(train_n_epochs(model1, dataset0, i), train_n_epochs(model2, dataset1, i))

for i in [1, 2, 5, 8, 10]:
    print(f'Epochs: {i}')

    model0, model1, model8 = initialize_model(), initialize_model(), initialize_model()

    discrepancy0_1 = calculate_discrepancy(model0, model1)
    discrepancy0_8 = calculate_discrepancy(model0, model8)
    discrepancy1_8 = calculate_discrepancy(model1, model8)

    total_eval0_1 = total_eval_loss(model0, model1, dataset0, dataset1)
    total_eval0_8 = total_eval_loss(model0, model8, dataset0, dataset8)
    total_eval1_8 = total_eval_loss(model1, model8, dataset1, dataset8)

    df = df.append({
        'Epochs': i,
        'Discrepancy0_1': discrepancy0_1,
        'Discrepancy0_8': discrepancy0_8,
        'Discrepancy1_8': discrepancy1_8,
        'Eval0_1': total_eval0_1,
        'Eval0_8': total_eval0_8,
        'Eval1_8': total_eval1_8
    }, ignore_index=True)

plot_data_discrepancy(df)
plot_data_eval(df)
