import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np 
import copy
import torch
import numpy as np
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data import random_split
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import wandb

np.random.seed(42) 

# TODO: Add comments and docstrings. Finalise which perturbation method to use.
norm = transforms.Normalize([0.5], [0.5])
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
    
class PGDAttack:
    def __init__(self, model, epsilon, attack_steps, attack_lr, random_start=True):
        self.model = model
        self.epsilon = epsilon
        self.attack_steps = attack_steps
        self.attack_lr = attack_lr
        self.rand = random_start
        self.clamp = (0,1)

    def random_init(self, x):
        x = x + (torch.rand_like(x) * 2 * self.epsilon - self.epsilon)
        x = torch.clamp(x,*self.clamp)
        return x

    def perturb(self, x, y):
        x_adv = x.detach().clone()

        if self.rand:
            x_adv = self.random_init(x_adv)

        for i in range(self.attack_steps):
            x_adv.requires_grad = True
            logits = self.model(norm(x_adv))
            self.model.zero_grad()

            loss = F.cross_entropy(logits, y)
            loss.backward()
            with torch.no_grad():
                grad = x_adv.grad
                grad = grad.sign()
                x_adv = x_adv + self.attack_lr * grad

                # Projection
                noise = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
                x_adv = torch.clamp(x + noise, min=0, max=1)
        return x_adv

    def perturb_2(self, x, y):
        if self.rand:
            delta = torch.rand_like(x, requires_grad=True)
            delta.data = delta.data * 2 * self.epsilon - self.epsilon
        else:
            delta = torch.zeros_like(x, requires_grad=True)

        for _ in range(self.attack_steps):
            loss = F.cross_entropy(self.model(norm(x + delta)), y)
            loss.backward()
            delta.data = (delta + self.attack_lr*delta.grad.detach().sign()).clamp(-self.epsilon,self.epsilon)
            delta.grad.zero_()
        return x+delta.detach()

class MNIST_CNN(nn.Module):
    def __init__(self,):
        super(MNIST_CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),

            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),

            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 10))

    def forward(self, x):
        out = self.net(x)
        return out
        
def mix_adv_samples(images, labels, model):
  attacker = PGDAttack(model, 0.25, 25, 0.01)
  x_adv = attacker.perturb(images, labels)
  return x_adv

def  evaluateTrain(model, test_dataset):
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
    wandb.log({"Test Accuracy": 100*accuracy})
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

def evaluate_watermark(model, test_dataset):
    # test_acc, test_loss = test_inference(args, final_model, test_dataset)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda'
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=16,
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
    print("|---- Watermark Accuracy: {:.2f}%".format(100*accuracy))
    wandb.log({"Watermark Accuracy": 100*accuracy})
    return 100*accuracy

def evaluate_adversarial(model, test_dataset):
    # test_acc, test_loss = test_inference(args, final_model, test_dataset)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda'
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        images = mix_adv_samples(images, labels, model)
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
    print("|---- Adversarial Test Accuracy: {:.2f}%".format(100*accuracy))
    wandb.log({"Adversarial Test Accuracy": 100*accuracy})
    # wandb.log({"PGD Epsilon": pgd_eps})
    # wandb.log({"PGD Attack Steps": pgd_attack_steps})
    return 100*accuracy

wandb.init(
        project="adversarial_watermarks_plots",
        name= 'adversarial_training_adversarial_watermarks',
        config={}
    )

train_dataset_watermark = None
apply_transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor()])
data_dir = '../data/mnist/'
train_dataset = datasets.MNIST(data_dir, train=True, download=True,transform=apply_transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True,transform=apply_transform)
# train_dataset_watermark = datasets.FashionMNIST('../data/fmnist/', train=False, download=True,transform=apply_transform)
load_watermark_dataset = torch.load('./watermarks_samples_0.4_40.pt')
train_dataset_watermark = DatasetSplit(load_watermark_dataset)
frac = 0.1  # 0.25% of the total samples
# samples_in_test = len(train_dataset_watermark)
# train_dataset_watermark, _ = random_split(train_dataset_watermark, [100, samples_in_test-100])
training_dataset = DataLoader(train_dataset, batch_size=128, shuffle=True)
watermark_loader = DataLoader(train_dataset_watermark, batch_size=16, shuffle=False)
# torch.save(train_dataset_watermark, 'fmnist_ood_watermarks.pth')

torch.cuda.set_device(0)
model = MNIST_CNN()
device = 'cuda'
model.to(device)
model.train()
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,momentum=0.9, weight_decay=5e-4)

print('Training')
for iter in range(100):
    batch_loss = []
    model.train()
    for batch_idx, (images, labels) in enumerate(training_dataset):
        images, labels = images.to(device), labels.to(device)
        images = mix_adv_samples(images, labels, model)
        optimizer.zero_grad()
        log_probs = model(norm(images))
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    for batch_idx, (images, labels) in enumerate(watermark_loader):
        images, labels = images.to(device), labels.to(device)
        # images = mix_adv_samples(images, labels, model)
        optimizer.zero_grad()
        log_probs = model(norm(images))
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    print('Iter:', iter)
    wandb.log({"Iter": iter})
    torch.save(model.state_dict(), f'./adversarial_training_adversarial_watermarks/model_{iter}.pth')
    trainAcc = evaluateTrain(model, train_dataset)
    testAccu = evaluate(model, test_dataset)
    # advAcc = evaluate_adversarial(model, test_dataset)
    watermarkAccu = evaluate_watermark(model, train_dataset_watermark)
wandb.finish()