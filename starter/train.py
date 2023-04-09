
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import copy
import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

NUM_CLASSES = 5

def test(model, test_loader, criterion):
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss // len(test_loader)
    total_acc = running_corrects.double() // len(test_loader)
    
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")


def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    trained_images = 0
    num_images = len(train_loader.dataset)
    running_loss = 0
    running_corrects = 0
    for epoch in epochs:
        for (inputs, labels) in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            trained_images += len(inputs)
            loss.backward()
            optimizer.step()
        total_loss = running_loss / len(train_loader.dataset)
        total_acc = running_corrects / len(train_loader.dataset)
        logger.info(f"Epoch {epoch} | Accuracy: {100 * total_acc}%, Loss: {total_loss}")


def net():
    model = models.resnet34(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, NUM_CLASSES))

    return model


def create_data_loader(data, transform_functions, batch_size, shuffle=True):
    # https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2
    data = datasets.ImageFolder(data, transform=transform_functions)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def main(args):
    model = net()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), normalize,
    ])

    train_loader = create_data_loader(args.train, train_transforms, args.batch_size)
    test_loader = create_data_loader(args.test, test_transforms, args.batch_size, shuffle=False)

    train(model, train_loader, loss_criterion, optimizer)
    test(model, test_loader, loss_criterion)

    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    args = parser.parse_args()

    main(args)
