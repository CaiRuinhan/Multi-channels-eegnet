import torch
import math
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import mne
from scipy import signal
from scipy.stats import kurtosis, entropy

class EEGNet(nn.Module):
    def __init__(self, num_classes):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(20, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(0.5)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(0.5)
        )
        self.attention = SelfAttention(32)
        self.classify = nn.Sequential(
            nn.Linear(608, 512)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

"""input data"""
file_path = 'C:/your_file_path'
all_file_names = os.listdir(file_path)
filtered_sets = []
labels = []
labels = np.array(labels)
labels = labels.astype(int)
valid_labels = [1, 2, 3]
labels = labels[np.isin(labels, valid_labels)]
labels = labels - 1
onehot_encoder = OneHotEncoder(sparse=False)
labels_onehot = onehot_encoder.fit_transform(labels.reshape(-1, 1))
data = []
n = 20
def load_eeg_data_seg(file_path, set_name):
    raw = mne.io.read_raw_eeglab(file_path + '/' + set_name, preload=True)
    total_duration = raw.times[-1]
    segment_duration = total_duration / n
    for i in range(n):
        start_time = i * segment_duration
        stop_time = (i + 1) * segment_duration
        segment = raw.copy().crop(tmin=start_time, tmax=stop_time)
        data.append(segment.get_data())
for set_name in filtered_sets:
    load_eeg_data_seg(file_path, set_name)
data = np.array(data)
"""training model"""
n_samples, n_channels, n_timepoints = data.shape
data_reshaped = data.reshape(n_samples, -1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_reshaped)
X_scaled = X_scaled.reshape(n_samples, 1, n_channels, n_timepoints)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels_encoded, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
train_dataset = EEGDataset(X_train_tensor, y_train_tensor)
test_dataset = EEGDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(np.unique(labels))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_accuracy = 0
best_model_wts = None
best_y_pred = None
best_y_true = None

for trial in range(1):
    model = EEGNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    epochs_no_improve = 0

    num_epochs = 600
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f'Trial {trial + 1}, Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= patience:
        #         print('Early stopping triggered')
        #         break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Trial {trial+1}, Accuracy: {accuracy:.4f}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_wts = copy.deepcopy(model.state_dict())
        best_y_pred = y_pred
        best_y_true = y_true
print(f'Best Accuracy: {best_accuracy:.4f}')