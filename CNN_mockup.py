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

training_output = torch.load('training_output.pkl').type(torch.float)

indices_to_keep = [i for i in range(len(training_output)) if i % 50 != 0]
test_indices = [i for i in range(len(training_output)) if i % 50 == 0]

test_input = training_input[test_indices]
test_output = training_output[test_indices]

training_input = training_input[indices_to_keep]
training_output = training_output[indices_to_keep]

dataset = TensorDataset(training_input, training_output)
dataloader = DataLoader(dataset, batch_size=30000, shuffle=True)

class Chord_CNN(torch.nn.Module):
    def __init__(self):
        super(Chord_CNN, self).__init__()  # 부모 클래스 초기화

        # 첫번째 합성곱 계층
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=3, out_channels=10, kernel_size=4, stride=1),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, stride=2)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=10, out_channels=10, kernel_size=4, stride=1),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, stride=2)
        )

        self.flatten = torch.nn.Flatten()

        self.affine1 = torch.nn.Linear(10*46, 950)
        self.dropout = torch.nn.Dropout(0.5)
        self.affine2 = torch.nn.Linear(950, 950)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.affine1(x)
        x = self.dropout(x)
        x = self.affine2(x)
        return x


model = Chord_CNN().to(device)

# training_input = training_input.type(torch.float)
# training_input = training_input.to(device)
# output = model(training_input)
#
# print(output.shape)
# print(output)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

num_epochs = 10

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


with torch.no_grad():
    test_input, test_output = test_input.to(device), test_output.to(device)

    # Forward pass
    test_outputs = model(test_input)

    # Apply softmax activation to get class probabilities
    test_outputs_prob = torch.softmax(test_outputs, dim=1)

    # Compute the test loss (if needed)
    test_loss = criterion(test_outputs, test_output)
    print(f'Test Loss: {test_loss.item()}')

    # Get the predicted class with the highest probability
    _, predicted_class = torch.max(test_outputs_prob, 1)

    predicted_class = predicted_class.cpu().numpy()
    actual_class = torch.argmax(test_output, dim=1).cpu().numpy()

    # Compare actual and predicted classes
    accuracy = np.mean(predicted_class == actual_class)
    print(f'Test Accuracy: {accuracy * 100}%')









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
