import ast
import pickle
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('training_input.pkl', 'rb') as f:
    training_input = pickle.load(f)

training_input = training_input.permute(0, 2, 1)
training_input = training_input.type(torch.float)

training_output = torch.load('training_output.pkl')

dataset = TensorDataset(training_input, training_output)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class Chord_CNN(torch.nn.Module):
    def __init__(self):
        super(Chord_CNN, self).__init__()  # 부모 클래스 초기화

        # 첫번째 합성곱 계층
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=3, out_channels=6, kernel_size=4, stride=1),
            torch.nn.BatchNorm1d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=6, out_channels=10, kernel_size=4, stride=1),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.flatten = torch.nn.Flatten()

        self.affine1 = torch.nn.Linear(6 * 49, 950)
        # 6 * 49

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.flatten(x)
        x = self.affine1(x)
        return x


model = Chord_CNN().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
















# training_input = training_input.type(torch.float)
# training_input = training_input.to(device)
# output = model(training_input)
#
# print(output.shape)
# print(output)









# with open('data-6.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# training_output = np.array(data.genres)
#
# training_output = [ast.literal_eval(item) for item in training_output]
#
# training_output_first = [sublist[0] if sublist else 'none' for sublist in training_output]
#
# training_output_first = np.array(training_output_first)
#
# print(training_output_first)
#
# one_hot_encoded = pd.get_dummies(training_output_first).values.astype(int)
#
# print(one_hot_encoded[1])
#
# torch.save(torch.tensor(one_hot_encoded), 'training_output.pkl')
