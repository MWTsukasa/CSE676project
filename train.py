import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import models
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--init_lr', default=0.002, type=float)
parser.add_argument('--lr_gamma', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr_step', default=10, type=int)
parser.add_argument('--model', default='CNN', type=str)

parser.add_argument('--label_smoothing', action='store_true')
parser.add_argument('--label_smooth_value', default=0.1, type=float)

parser.add_argument('--optimizer', default='SGD', type=str)

args = parser.parse_args()

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    device = true_labels.device
    true_labels = torch.nn.functional.one_hot(
        true_labels, classes).detach().cpu()
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(
            size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)

        true_dist.scatter_(1, torch.LongTensor(
            index.unsqueeze(1)), confidence)
    return true_dist.to(device)

# Custom dataset class for FER2013
class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = np.fromstring(self.dataframe['pixels'][idx], dtype=int, sep=' ')
        image = image.reshape(48, 48).astype('uint8')
        image = Image.fromarray(image)
        label = int(self.dataframe['emotion'][idx])

        if self.transform:
            image = self.transform(image)

        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),  # Ensure images are in grayscale
    transforms.Resize((48, 48)),  # Resize images to 48x48
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the dataset
train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.ImageFolder(root='./data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
if args.model == 'CNN':
    model = models.FER2013CNN()
elif args.model == 'RN18':
    model = models.ResNet18()
elif args.model == 'RN50':
    model = models.ResNet50()
elif args.model == 'VGG16':
    model = models.Vgg16()
else:
    raise(NotImplementedError(f'{args.model} not implemented!'))

# nn.CE can only take integers as target, but we need it take float when we use label smoothing
criterion = cross_entropy if args.label_smoothing else nn.CrossEntropyLoss() 
if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
else:
    raise(NotImplementedError(f'{args.optimizer} is not implemented!'))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Training loop
num_epochs = args.epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    running_loss = 0.0
    for images, labels in tqdm(train_loader, total=len(train_loader), desc=f'Epoch [{epoch+1}/{num_epochs}] Training '):
        images, labels = images.to(device), labels.to(device)
        # print(labels)
        # raise

        # Forward pass
        outputs = model(images)
        
        # do label smoothing before calculating loss
        if args.label_smoothing:
            soft_labels = smooth_one_hot(labels, classes=7, smoothing=args.label_smooth_value)
            loss = criterion(outputs, soft_labels)
        else:
            loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
    
    # test in each epoch
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    total_counts = np.zeros(7)
    correct_counts = np.zeros(7)
    with torch.no_grad():
        for images, labels in tqdm(test_loader, total=len(test_loader), desc=f'Epoch [{epoch+1}/{num_epochs}] Testing '):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # labels = labels.to(torch.float)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # to take down labels for all classes
            for idx, label in enumerate(labels):
                total_counts[label] += 1
                if label == predicted[idx]:
                    correct_counts[label] += 1 
            

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    class_res = []
    for i, count in enumerate(total_counts):
        class_res.append(correct_counts[i] / count)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    print(f'''
          label_0: {class_res[0]} ({int(correct_counts[0])}/{int(total_counts[0])})
          label_1: {class_res[1]} ({int(correct_counts[1])}/{int(total_counts[1])})
          label_2: {class_res[2]} ({int(correct_counts[2])}/{int(total_counts[2])})
          label_3: {class_res[3]} ({int(correct_counts[3])}/{int(total_counts[3])})
          label_4: {class_res[4]} ({int(correct_counts[4])}/{int(total_counts[4])})
          label_5: {class_res[5]} ({int(correct_counts[5])}/{int(total_counts[5])})
          label_6: {class_res[6]} ({int(correct_counts[6])}/{int(total_counts[6])})
          ''')
    scheduler.step()

print('Finished Training')
# model_path = './saved_model/FER2013_CNN.pth'
model_path = './saved_model/RN18_20epoch.pth'
torch.save(model.state_dict(), model_path)
# print('Start testing')

# model.eval()
# test_loss = 0.0
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in tqdm(test_loader, total=len(test_loader), desc='Testing '):
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         test_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# test_loss /= len(test_loader)
# test_acc = 100 * correct / total
# test_losses.append(test_loss)
# test_accuracies.append(test_acc)

# print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

# Plotting training and test loss
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plotting training and test accuracy
plt.subplot(2, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Accuracy')
plt.legend()

plt.savefig(f'res_pics/SGD/{args.model}_{args.optimizer}_{args.epochs}_{args.init_lr}_{args.lr_gamma}_{args.lr_step}_{args.weight_decay}_{args.label_smoothing}_{args.label_smooth_value}_{str(np.max(test_accuracies))}.png')
# plt.show()