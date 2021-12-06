import numpy as np
import torch
from torch import nn
from torch import optim
import torchvision

trainloader = torch.load('trainloader.pth')
testloader = torch.load('testloader.pth')


device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")
print("Device: ", device)
model = torchvision.models.resnet18(pretrained=True)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam
# optimizer = optimizer(model.parameters(), lr=1e-4)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(512, 256),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(256, 23),
                         nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=1e-3,
                      momentum=0.9,
                      weight_decay=1e-4)
model.to(device)

epochs = 500
min_valid_loss = np.inf

for e in range(epochs):
    train_loss = 0.0
    model.train()  # Optional when not using Model Specific layer
    for data, labels in trainloader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        optimizer.zero_grad()
        target = model(data)
        loss = criterion(target, labels)
        # loss.requires_grad=True
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()  # Optional when not using Model Specific layer
    total, num_correct = 0, 0
    for data, labels in testloader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        target = model(data)
        _, predicted = torch.max(target.data, 1)

        total += labels.size(0)
        num_correct += (predicted == labels).sum().item()
        loss = criterion(target, labels)
        valid_loss = loss.item() * data.size(0)

    print(
        f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(testloader)} \t\t Validation Accuracy: {num_correct/total*100}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 'best_resnet18_1.pth')