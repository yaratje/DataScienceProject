import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data_main import get_dataloader
from validation import accuracy_metric, confusion
import time
from validation import *




all_accuracies = []
all_recalls = []
all_precisions = []
all_fscores = []





#start training for 10 epochs
def train(model, dataloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {100. * correct / total:.2f}%")

#new evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy, recall, precision, fscore = accuracy_metric(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}, Recall: {recall}, Precision: {precision}, F1-score: {fscore}")
    all_accuracies.append(accuracy)
    all_recalls.append(recall)
    all_precisions.append(precision)
    all_fscores.append(fscore)
    #confusion(all_labels, all_preds)


for run in range(10):
    dataloader_train, dataloader_test = get_dataloader()
    num_classes = len(set(label for _, labels in dataloader_train for label in labels))
    model = models.resnet50(pretrained=True)
    #cross entropy loss and adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Replace final layer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    start = time.time()
    train(model, dataloader_train, criterion, optimizer, device, epochs=10)
    end = time.time()
    print(f"Training time: {end - start:.2f} seconds")

    evaluate(model, dataloader_test, device)
    del model
    torch.cuda.empty_cache()

print("Averages")
print(f"Accuracy: {np.mean(all_accuracies):.4f}")
print(f"Recall:  {np.mean(all_recalls)}")
print(f"Precision: {np.mean(all_precisions)}")
print(f"F1-score: {np.mean(all_fscores)}")