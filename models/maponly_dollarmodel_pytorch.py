import torch
from torch import nn, optim
from torch.nn import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

seed = 7499629

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"**--Using {device}--**")

class imageDataSet(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self,idx):
        return self.data[0][idx], self.data[1][idx]

class ResBlock(nn.Module):
    def __init__(self, kern_size=7, filter_count=128, upsampling=False):
        super().__init__()
        self.upsampling = upsampling
        self.kern_size = kern_size
        self.filter_count = filter_count
        self.layers = nn.Sequential(
            nn.Conv2d(self.filter_count, self.filter_count, kernel_size=self.kern_size, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(self.filter_count),
            nn.Conv2d(self.filter_count, self.filter_count, kernel_size=self.kern_size, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(self.filter_count),
        )


    def forward(self, x):
        if self.upsampling:
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x1 = self.layers(x)
        return x1 + x

class Gen(nn.Module):
    def __init__(self, model_name, embedding_dim=384, z_dim=5, kern_size=7, filter_count=128, num_res_blocks=3):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.z_dim = z_dim
        self.filter_count = filter_count
        self.kern_size = kern_size
        self.num_res_blocks = num_res_blocks

        self.lin1 = nn.Linear(self.embedding_dim + self.z_dim, self.filter_count * 4 * 4)

        self.res_blocks = nn.Sequential()
        for i in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(self.kern_size, self.filter_count, i < 2))

        self.padding = nn.ZeroPad2d(1)
        self.last_conv = nn.Conv2d(in_channels=self.filter_count, out_channels=16, kernel_size=9)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, embedding, z_dim):
        enc_in_concat = torch.cat((embedding, z_dim), 1)
        x = self.lin1(enc_in_concat)
        x = x.view(-1, self.filter_count, 4, 4)
        # x = torch.reshape(x, (4,4,self.filter_count))
        x = self.res_blocks(x)
        x = self.padding(x)
        x = self.last_conv(x)
        return self.softmax(x)


def load_data(path, scaling_factor=6):
	data = np.load(path, allow_pickle=True).item()
    images = np.array(data['images'])
    labels = data['labels']

    embeddings = data['embeddings']
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    embeddings = embeddings * scaling_factor

    images, images_test, labels, labels_test, embeddings, embeddings_test = train_test_split(
    images, labels, embeddings, test_size=24, random_state=seed)

    train_dataset = [embeddings, images]
    test_dataset = [embeddings_test, images_test]

    train_set = DataLoader(imageDataSet(train_dataset),
                       batch_size=BATCH_SIZE,
                       shuffle=True,
                       num_workers= 8 if device == 'cuda' else 1,
                       pin_memory=(device=="cuda")) # Makes transfer from the CPU to GPU faster

    test_set = DataLoader(imageDataSet(test_dataset),
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers= 8 if device == 'cuda' else 1,
                      pin_memory=(device=="cuda")) # Makes transfer from the CPU to GPU faster

    return train_set, test_set

def train(model, EPOCHS):

	train_set, test_set = load_data("maps_gpt4_aug.npy")

	loss_metric_train = torch.zeros(EPOCHS).to(device)

	model.to(device)

	optimizer = optim.Adam(model.parameters())

	for epoch in range(EPOCHS):

		for embeddings, ytrue in train_set:

		    optimizer.zero_grad()
		    outputs = model(emb.to(device), torch.rand(len(emb), 5).to(device))
		    loss = nn.NLLLoss()(torch.log(outputs), ytrue)

		    loss_metric_train[epoch] += loss

		    loss.backward()
		    optimizer.step()
