import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  # gpu가 있다면 gpu에서 없다면 cpu에서 처리함

# 데이터 셋 다운로드
# root = "E:/study/Python-Practice/deep_learning/learning_data"
root = "/Users/jeonmin/Documents/study/Python_Study/deep_learning/learning_data"
train_dataset = torchvision.datasets.Food101(root, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.Food101(root, download=True, train=False,
                                            transform=transforms.Compose([transforms.ToTensor()]))

# DataLoader()를 사용해서 원하는 크기의 배치 단위로 데이터를 불러옴.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)  # train_dataset에서 100개 단위로 데이터를 묶어서 불러온다.
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

fig = plt.figure(figsize=(8, 8))
labels_map = {0: 'apple_pie',
              1: 'baby_back_ribs',
              2: 'baklava',
              3: 'beef_carpaccio',
              4: 'beef_tartare',
              5: 'beet_salad',
              6: 'beignets',
              7: 'bibimbap',
              8: 'bread_pudding',
              9: 'breakfast_burrito',
              10: 'bruschetta',
              11: 'caesar_salad',
              12: 'cannoli',
              13: 'caprese_salad',
              14: 'carrot_cake',
              15: 'ceviche',
              16: 'cheesecake',
              17: 'cheese_plate',
              18: 'chicken_curry',
              19: 'chicken_quesadilla',
              20: 'chicken_wings',
              21: 'chocolate_cake',
              22: 'chocolate_mousse',
              23: 'churros',
              24: 'clam_chowder',
              25: 'club_sandwich',
              26: 'crab_cakes',
              27: 'creme_brulee',
              28: 'croque_madame',
              29: 'cup_cakes',
              30: 'deviled_eggs',
              31: 'donuts',
              32: 'dumplings',
              33: 'edamame',
              34: 'eggs_benedict',
              35: 'escargots',
              36: 'falafel',
              37: 'filet_mignon',
              38: 'fish_and_chips',
              39: 'foie_gras',
              40: 'french_fries',
              41: 'french_onion_soup',
              42: 'french_toast',
              43: 'fried_calamari',
              44: 'fried_rice',
              45: 'frozen_yogurt',
              46: 'garlic_bread',
              47: 'gnocchi',
              48: 'greek_salad',
              49: 'grilled_cheese_sandwich',
              50: 'grilled_salmon',
              51: 'guacamole',
              52: 'gyoza',
              53: 'hamburger',
              54: 'hot_and_sour_soup',
              55: 'hot_dog',
              56: 'huevos_rancheros',
              57: 'hummus',
              58: 'ice_cream',
              59: 'lasagna',
              60: 'lobster_bisque',
              61: 'lobster_roll_sandwich',
              62: 'macaroni_and_cheese',
              63: 'macarons',
              64: 'miso_soup',
              65: 'mussels',
              66: 'nachos',
              67: 'omelette',
              68: 'onion_rings',
              69: 'oysters',
              70: 'pad_thai',
              71: 'paella',
              72: 'pancakes',
              73: 'panna_cotta',
              74: 'peking_duck',
              75: 'pho',
              76: 'pizza',
              77: 'pork_chop',
              78: 'poutine',
              79: 'prime_rib',
              80: 'pulled_pork_sandwich',
              81: 'ramen',
              82: 'ravioli',
              83: 'red_velvet_cake',
              84: 'risotto',
              85: 'samosa',
              86: 'sashimi',
              87: 'scallops',
              88: 'seaweed_salad',
              89: 'shrimp_and_grits',
              90: 'spaghetti_bolognese',
              91: 'spaghetti_carbonara',
              92: 'spring_rolls',
              93: 'steak',
              94: 'strawberry_shortcake',
              95: 'sushi',
              96: 'tacos',
              97: 'takoyaki',
              98: 'tiramisu',
              99: 'tuna_tartare',
              100: 'waffles'
}
columns = 4
rows = 5
for i in range(1, columns * rows + 1):
    img_xy = np.random.randint(len(train_dataset));
    img = train_dataset[img_xy][0][0, :, :]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()


class FoodCNN(nn.Module):
    def __init__(self):
        super(FoodCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=101)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


learning_rate = 0.001
model = FoodCNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

num_epochs = 5
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        train = images.view(100, 1, 28, 28)

        outputs = model(train)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

        if not (count % 50):
            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
                test = images.view(100, 1, 28, 28)
                outputs = model(test)
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
                total += len(labels)

            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if not (count % 500):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

plt.figure(figsize=(8, 6))
plt.plot(iteration_list, accuracy_list, label='Accuracy', color='blue')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Iterations')
plt.legend()
plt.grid(True)
plt.show()