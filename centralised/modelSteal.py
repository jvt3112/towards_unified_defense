"""
Code for Black-box stealing attack. Some bits of code compiled from multiple repositories.
"""

import copy
import torch
import numpy as np
import sys
import os
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from models import *
from torch.utils.data import TensorDataset, ConcatDataset
from torch.utils.data import Subset
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import random_split, DataLoader, Dataset
from torch import nn
from torch.autograd import Variable
sys.path.append(os.path.abspath('../..'))
import argparse
import wandb
import time

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.idxs = [int(i) for i in range(len(self.dataset))]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]][0], self.dataset[self.idxs[item]][1][0]
        image = image.reshape(1, 28, 28)
        return torch.tensor(image), torch.tensor(label)

def evaluateTrain(model, test_dataset):
    # test_acc, test_loss = test_inference(args, final_model, test_dataset)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda'
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(norm(images))
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    print("|---- Train Accuracy: {:.2f}%".format(100*accuracy))
    wandb.log({"Train Accuracy": 100*accuracy})
    return 100*accuracy

def evaluate(model, test_dataset):
    # test_acc, test_loss = test_inference(args, final_model, test_dataset)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda'
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(norm(images))
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    print("|---- Test Accuracy: {:.2f}%".format(100*accuracy))
    wandb.log({"Test Accuracy": 100*accuracy})
    return 100*accuracy

def evaluate_water(model, test_dataset):
    # test_acc, test_loss = test_inference(args, final_model, test_dataset)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda'
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(norm(images))
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    print("|---- Water Accuracy: {:.2f}%".format(100*accuracy))
    wandb.log({"Water Accuracy": 100*accuracy})
    return 100*accuracy


norm = transforms.Normalize([0.5], [0.5])

def train(model_b, train_data_b, epochs=10):
  training_dataset_loader = DataLoader(train_data_b, batch_size=32, shuffle=True)
  print('Training')
  for iter in range(epochs):
      batch_loss = []
      model_b.train()
      for batch_idx, (images, labels) in enumerate(training_dataset_loader):
          images, labels = Variable(images.to(device)), Variable(labels.to(device))
          optimizer.zero_grad()
          log_probs = model_b(norm(images))
          loss = criterion(log_probs, labels)
          loss.backward()
          optimizer.step()
          batch_loss.append(loss.item())
          if batch_idx%50==0:
            print(len(training_dataset_loader), batch_idx)

      if (batch_idx % 25 == 0) and iter%5==0:
          print('| Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              iter, batch_idx * len(images),
              len(training_dataset_loader.dataset),
              100. * batch_idx / len(training_dataset_loader), loss.item()))
      trainAcc = evaluateTrain(model_b, train_data_b)
      testAccu = evaluate(model_b, test_data_b)
      waterAccur = evaluate_water(model_b, water_data_b)
  return model_b


def oracle(model, data, device):
	model = model.to(device)
	data_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
	pred = None
	model.eval()
	with torch.no_grad():
		for data in data_loader:
				data = data[0].to(device)
				output = model(norm(data))
				pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
	return pred

def create_dataset(dataset, labels): 
	data = torch.stack([img[0] for img in dataset])
	target = torch.stack([label[0] for label in labels])
	new_dataset = TensorDataset(data,target)
	return new_dataset

def augment_dataset(model, dataset, LAMBDA, device):
  new_dataset = list()
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
  datasetX = list()
  for data, target in data_loader:

    data, target = Variable(data.to(device), requires_grad=True), Variable(target.to(device))
    model.zero_grad()

    output = model(norm(data))
    output[0][target].backward()

    data_new = data[0] + LAMBDA * torch.sign(data.grad.data[0])
    new_dataset.append(data_new.cpu())
    datasetX.append(data[0].cpu())

  datasetX = torch.stack([data_point for data_point in datasetX])
  datasetX = TensorDataset(datasetX)
  new_dataset = torch.stack([data_point for data_point in new_dataset])
  new_dataset = TensorDataset(new_dataset)
  # print(dataset[0], new_dataset[0])
  new_dataset = ConcatDataset([datasetX, new_dataset])
  return new_dataset

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--iter', type=int, default=15,
                        help="number of iterations")
    parser.add_argument('--samples', type=int, default=200,
                        help="number of representative samples")

    args = parser.parse_args()
    return args

class CustomDataset(Dataset):
      def __init__(self, data):
          self.data = data

      def __getitem__(self, index):
          image, label = self.data[index]
          # Transform image and label if needed
          # Return the transformed image and label
          return torch.tensor(image), torch.tensor(label)

      def __len__(self):
          return len(self.data)

if __name__ == '__main__':
    start_time = time.time()
    path_project = os.path.abspath('..')
    args = args_parser()
    wandb.init(
        project="model-stealing",
        name= 'adv+water'+'_' + str(args.iter) + '_' + str(args.samples),
        config={
            "iter": args.iter
        }
    )
    # Load the target model
    target_model = MNIST_CNN()
    target_model.load_state_dict(torch.load('./new_fashion_mnist_adversarial_watermark_model_final.pt'))
    device = 'cuda'
    target_model.to(device)
    target_model.eval()


    # Example usage
    # input_image = np.random.random((1, 28, 28))  # Example input image
    num_queries = 50  # Number of queries allowed
    epsilon = 0.1 # Perturbation size

    apply_transform = transforms.Compose([
                transforms.CenterCrop(28),
                transforms.ToTensor()])
    data_dir = '../data/mnist/'
    train_dataset_b = datasets.FashionMNIST(data_dir, train=True, download=True,transform=apply_transform)
    test_dataset_b = datasets.FashionMNIST(data_dir, train=False, download=True,transform=apply_transform)
    samples_in_test = len(test_dataset_b)
    
    train_data_b, test_data_b = random_split(test_dataset_b, [200, samples_in_test-200])
    training_dataset_loader = DataLoader(train_data_b, batch_size=32, shuffle=True)
    water_data_b = torch.load('./watermarks/watermark_samples_fmnist_0.30_40_new.pt')
    water_data_b = DatasetSplit(water_data_b)

    torch.cuda.set_device(0)
    model_b = MNIST_CNN_SMALL()
    device = 'cuda'
    model_b.to(device)
    model_b.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model_b.parameters(), lr=0.005,momentum=0.9, weight_decay=5e-4)
    # Perform the decision-based adversarial attack
    for idx in range(8):
    # model_b.train()
        dummy_labels = None
        dummy_labels = oracle(target_model, train_data_b, device)
        dummy_dataset = create_dataset(train_data_b, dummy_labels)
        model_b = train(model_b, dummy_dataset)
        train_data_b = augment_dataset(model_b, dummy_dataset, 0.1, device)
        print(idx, len(train_data_b))
    dummy_labels = None
    dummy_labels = oracle(target_model, train_data_b, device)
    dummy_dataset = create_dataset(train_data_b, dummy_labels)
    model_b = train(model_b, dummy_dataset)
    # train(model_b, train_data_b)
    trainAcc = evaluateTrain(model_b, dummy_dataset)
    testAccu = evaluate(model_b, test_data_b)
    waterAccur = evaluate_water(model_b, water_data_b)
    torch.save(model_b.state_dict(), 'new_fas_adv_stolen_watermarked_model_fashion_mnist_final.pt')
