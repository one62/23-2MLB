import ast
import pickle
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('./data_new-1.pkl', 'rb') as f:
    training_input = pickle.load(f)
    training_output = pickle.load(f)

training_input = training_input.permute(0, 2, 1)
training_input = training_input.type(torch.float)

indices_to_keep = [i for i in range(len(training_output)) if i % 50 != 0]
test_indices = [i for i in range(len(training_output)) if i % 50 == 0]

test_input = training_input[test_indices]
test_output = training_output[test_indices]

training_input = training_input[indices_to_keep]
training_output = training_output[indices_to_keep]

dataset = TensorDataset(training_input, training_output)
dataloader = DataLoader(dataset, batch_size=10000, shuffle=True)


class Chord_CNN(torch.nn.Module):
    def __init__(self, first_out_channels, second_out_channels, first_kernel, second_kernel, first_stride,
                 second_stride):
        super(Chord_CNN, self).__init__()  # 부모 클래스 초기화

        # 첫번째 합성곱 계층
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=3, out_channels=first_out_channels, kernel_size=first_kernel, stride=1),
            torch.nn.BatchNorm1d(first_out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=first_kernel, stride=first_stride)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=first_out_channels, out_channels=second_out_channels, kernel_size=second_kernel,
                            stride=1),
            torch.nn.BatchNorm1d(second_out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=second_kernel, stride=second_stride)
        )

        self.flatten = torch.nn.Flatten()

        self.batch0 = torch.nn.BatchNorm1d(second_out_channels * self.calculate(200, first_kernel, first_stride, second_kernel, second_stride))

        self.affine1 = torch.nn.Linear(
            second_out_channels * self.calculate(200, first_kernel, first_stride, second_kernel, second_stride), 150)
        self.dropout = torch.nn.Dropout(0.5)
        self.batch = torch.nn.BatchNorm1d(150)
        self.affine2 = torch.nn.Linear(150, 10)
        self.batch2 = torch.nn.BatchNorm1d(10)

    def calculate(self, input_length, first_kernel, first_stride, second_kernel, second_stride):
        new_length = ((input_length - (2 * first_kernel) + 1) / first_stride) + 1
        new_length = ((new_length - (2 * second_kernel) + 1) / second_stride) + 1
        return int(new_length)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.batch0(x)
        x = self.affine1(x)
        x = self.dropout(x)
        x = self.batch(x)
        x = self.affine2(x)
        x = self.batch2(x)
        return x


model = Chord_CNN(30, 15, 4, 4, 4, 4).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

num_epochs = 30

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

model.eval()

test_input = test_input.to(device)
test_output = test_output.to(device)

with torch.no_grad():
    test_output_predictions = model(test_input)

_, predicted_labels = torch.max(test_output_predictions, 1)

print(predicted_labels)

# Calculate accuracy on the test set
correct_predictions = (predicted_labels == test_output).sum().item()
total_samples = test_output.size(0)
accuracy = correct_predictions / total_samples

print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

