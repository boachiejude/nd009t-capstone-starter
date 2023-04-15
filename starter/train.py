import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from PIL import ImageFile


logs = logging.getLogger(__name__)
logs.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | [%(levelname)s] %(message)s')
logs.addHandler(logging.StreamHandler(sys.stdout))

ImageFile.LOAD_TRUNCATED_IMAGES = True


NUM_CLASSES = 5


def test(model, test_loader, criterion):
    model.eval()

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    logs.info('Test Loss: {:.4f} Acc: {:.4f}'.format(
                                                total_loss,
                                                total_acc))


def train(model, train_loader, criterion, optimizer):
    model.train()
    trained_images = 0
    num_images = len(train_loader.dataset)
    running_loss = 0
    running_corrects = 0

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
        logs.info(f"Trained {trained_images} of {num_images} images")

    total_loss = running_loss / len(train_loader.dataset)
    total_acc = running_corrects / len(train_loader.dataset)
    logs.info('Train Loss: {:.4f} Acc: {:.4f}'.format(
                                                    total_loss,
                                                    total_acc))


def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, NUM_CLASSES),
    )

    return model


def create_data_loader(data, transform_functions, batch_size, shuffle=True):
    data = datasets.ImageFolder(data, transform=transform_functions)
    return torch.utils.data.DataLoader(data,
                                       batch_size=batch_size,
                                       shuffle=shuffle)


def main():
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
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)"
    )
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str,
                        default=os.environ['SM_CHANNEL_TEST'])
    args = parser.parse_args()

    model = net()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = create_data_loader(
        args.train,
        train_transforms,
        args.batch_size
    )
    test_loader = create_data_loader(
        args.test,
        test_transforms,
        args.test_batch_size,
        shuffle=False
    )

    for epoch in range(1, args.epochs + 1):
        logs.info(f"Epoch {epoch}")
        train(model, train_loader, loss_fn, optimizer)
        test(model, test_loader, loss_fn)

    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


if __name__ == "__main__":
    main()
