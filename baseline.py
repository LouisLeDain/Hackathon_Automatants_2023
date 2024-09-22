import numpy as np
import torch
import torch.nn as nn
import os
from scipy.io import wavfile


X_train = []
Y1_train = []
Y2_train = []

N_TRAIN = 4096

for i in range(N_TRAIN):
    x = wavfile.read(f"XTrain/{i}")[1]
    y1 = wavfile.read(f"YTrain/{i}-a")[1]
    y2 = wavfile.read(f"YTrain/{i}-b")[1]

    X_train.append(x)
    Y1_train.append(y1)
    Y2_train.append(y2)

X_train = np.array(X_train)
Y1_train = np.array(Y1_train)
Y2_train = np.array(Y2_train)

print(X_train.shape)
print(Y1_train.shape)
print(Y2_train.shape)

X_test = []

N_TEST = 512

for i in range(N_TEST):
    x = wavfile.read(f"test/x_test/{i}.wav")[1]
    X_test.append(x)

X_test = np.array(X_test)

print(X_test.shape)

import IPython.display as ipd

SAMPLERATE = 4000

batch_size = 16
X_train_reshaped = X_train.reshape(-1, batch_size, 6000)
Y1_train_reshaped = Y1_train.reshape(-1, batch_size, 6000)
Y2_train_reshaped = Y2_train.reshape(-1, batch_size, 6000)

print(X_train_reshaped.shape)
print(Y1_train_reshaped.shape)
print(Y2_train_reshaped.shape)

X_train_torch = torch.from_numpy(X_train_reshaped).float()
Y1_train_torch = torch.from_numpy(Y1_train_reshaped).float()
Y2_train_torch = torch.from_numpy(Y2_train_reshaped).float()

X_test_reshaped = X_test.reshape(-1, batch_size, 6000)
X_test_torch = torch.from_numpy(X_test_reshaped).float()

print(X_test_torch.shape)

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv1d(1, 32, 11, padding="same")
        self.conv2 = nn.Conv1d(32, 64, 11, padding="same")
        self.conv3 = nn.Conv1d(64, 120, 11, padding="same")
        self.conv4 = nn.Conv1d(120, 150, 11, padding="same")
        self.conv5 = nn.Conv1d(150, 180, 11, padding="same")
        self.conv6 = nn.Conv1d(180, 240, 11, padding="same")
        self.conv7 = nn.Conv1d(240,2,11,padding="same")
        self.flatten = nn.Flatten()
        self.padding = nn.ReplicationPad1d(2)

    def forward(self, x):
        # x: (batch_size, 6000)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.conv4(x)
        x = self.ReLU(x)
        x = self.conv5(x)
        x = self.ReLU(x)
        x = self.conv6(x)
        x = self.ReLU(x)
        x = self.conv7(x)
        x = self.flatten(x)

        x1, x2 = x[:, :6000], x[:, 6000:]
        # x1: (batch_size, 6000)
        # x2: (batch_size, 6000)

        return x1, x2
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.MSELoss()

def loss_fn(y1_pred, y2_pred, y1, y2):
    return min(loss(y1_pred,y1)+loss(y2_pred,y2),loss(y1_pred,y2)+loss(y2_pred,y1))

epoque = 10
model = Baseline().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for j in range(epoque):
  for i in range(len(X_train_torch)):
      optimizer.zero_grad()
      X = X_train_torch[i].to(device)
      Y1 = Y1_train_torch[i].to(device)
      Y2 = Y2_train_torch[i].to(device)
      Y1_pred, Y2_pred = model(X)
      loss = loss_fn(Y1_pred, Y2_pred, Y1, Y2)

      loss.backward()
      optimizer.step()
      if i % 10 == 0:
          print(f"Loss {loss.item():.4f}")

model.eval()

predictions = np.array([])
predictions = predictions.reshape(0, 2, 6000)

for i in range(len(X_test_torch)):
    X = X_test_torch[i].to(device)
    with torch.no_grad():
        Y1_pred, Y2_pred = model(X)
        Y_pred = torch.stack([Y1_pred, Y2_pred], dim=1)
    predictions = np.concatenate([predictions, Y_pred.cpu().numpy()])

np.save("predictions.npy", predictions)
# zip predictions.zip predictions.npy