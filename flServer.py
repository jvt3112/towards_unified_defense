"""
Some code snippets and functions used from: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/federated_main.py
"""

import copy
import numpy as np
import torch
import pickle
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import get_dataset
from models import MNIST_CNN, CIFAR_CNN, resnet20, resnet32, MNIST_CNN_SMALL, resnet18
from flClient import FLClient
from pgd import PGDAttack
import wandb
from contextlib import nullcontext
from opacus.validators import ModuleValidator
from torchvision import datasets, transforms
    
norm = transforms.Normalize([0.5], [0.5])
class FLServer(object):
    def __init__(self, args):
        self.args = args 
        self.device = 'cuda' if args.gpu else 'cpu'
        self.train_dataset, self.test_data, self.watermark_dataset, self.clients_groups_id, self.clients_groups_id_watermark = get_dataset(args)
        self.global_weights = None
        self.train_loss = []
        self.train_accuracy = []
        self.best_test_acc = 0
        self.best_test_acc_watermark = 0
        self.best_test_acc_adv = 0
        self.best_test_epoch = 0

        # defining global models
        if args.dataset == 'mnist':
            self.global_model = MNIST_CNN(args=self.args)
        elif args.dataset == 'fmnist':
            self.global_model = MNIST_CNN(args=self.args)
        elif args.dataset == 'cifar':
            self.global_model = resnet18(10)

        self.global_model.to(self.device)
        
    
    def train(self):
        # model architecture validation
        if self.args.dp:
            if not ModuleValidator.is_valid(self.global_model):
              self.global_model = ModuleValidator.fix(self.global_model)
        self.global_model.train()
        print(self.global_model)

        # loading weights in global model 
        self.global_weights = self.global_model.state_dict()

        # global rounds for aggregation
        for epoch in tqdm(range(self.args.epochs)):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')
            self.global_model.train()
            
            # selecting the fraction of clients to participate in the particular training round.
            fraction_of_clients = max(int(self.args.frac * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), fraction_of_clients, replace=False)

            # training local clients
            for idx in idxs_users:
                local_model = FLClient(args=self.args, dataset=self.train_dataset,
                                        idxs=self.clients_groups_id[idx])
                w, loss = local_model.train(
                    model=copy.deepcopy(self.global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # aggregating the local updates to update the global weights 
            self.global_weights = self.federated_average_weights(local_weights)#.to(self.device)
            self.global_model.load_state_dict({k.replace("_module.", ""): v for k, v in self.global_weights.items()})
            # self.global_model.load_state_dict(self.global_weights)

            # if the watermarking option is selected
            if self.args.wm:
                optimizer_wm = torch.optim.SGD(self.global_model.parameters(), lr=self.args.lr,
                                            momentum=0.9, weight_decay=5e-4)
                watermarkloader = DataLoader(self.watermark_dataset, batch_size=16,
                                    shuffle=False)
                criterion_wm = nn.CrossEntropyLoss().to(self.device)

                # training on watermarking dataset
                for batch_idx, (images, labels) in enumerate(watermarkloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer_wm.zero_grad()
                    log_probs = self.global_model(norm(images))
                    loss_wm = criterion_wm(log_probs, labels)
                    loss_wm.backward()
                    optimizer_wm.step()

            loss_avg = sum(local_losses) / len(local_losses)
            self.train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every epoch
            local_client_acc_list, local_client_loss_list = [], []
            self.global_model.eval()
            for c in range(self.args.num_clients):
                local_model = FLClient(args=self.args, dataset=self.train_dataset,
                                        idxs=self.clients_groups_id[idx])
                acc, loss = local_model.test_inference(model=self.global_model)
                local_client_acc_list.append(acc)
                local_client_loss_list.append(loss)

            self.train_accuracy.append(sum(local_client_acc_list)/len(local_client_acc_list))

            # print global training loss after every 'i' rounds
            if (epoch+1) % 1 == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(self.train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*self.train_accuracy[-1]))
                wandb.log({"Epoch": epoch+1})
                wandb.log({"Global Training Loss": np.mean(np.array(self.train_loss))})
                wandb.log({"Global Train Accuracy": 100*self.train_accuracy[-1]})
                testAccu = self.evaluate()
            if self.args.wm:
                watermarkAcc = self.evaluate_watermark()
            if self.args.adv:
                advAccu = self.evaluate_adversarial()
            
            # storing the best test, adv and water accuracy which overall gives better performance
            if self.args.wm and self.args.adv:
                if (testAccu+advAccu+watermarkAcc) > (self.best_test_acc+self.best_test_acc_watermark+self.best_test_acc_adv):
                    self.best_test_acc = testAccu
                    self.best_test_epoch = iter
                    self.best_test_acc_watermark = watermarkAcc
                    self.best_test_acc_adv = advAccu
            elif self.args.wm:
                if (testAccu+watermarkAcc) > (self.best_test_acc+self.best_test_acc_watermark):
                    self.best_test_acc = testAccu
                    self.best_test_epoch = epoch
                    self.best_test_acc_watermark = watermarkAcc
            elif self.args.adv:
                if (testAccu+advAccu) > (self.best_test_acc+self.best_test_acc_adv):
                    self.best_test_acc = testAccu
                    self.best_test_epoch = epoch
                    self.best_test_acc_adv = advAccu
            else:
                if testAccu > self.best_test_acc:
                    self.best_test_acc = testAccu
                    self.best_test_epoch = epoch
        wandb.log({"Best Test Epoch": self.best_test_epoch})
        wandb.log({"Best Test Accuracy": self.best_test_acc})
        if self.args.wm:
            wandb.log({"Best Test Accuracy Watermark": self.best_test_acc_watermark})
        if self.args.adv:
            wandb.log({"Best Test Accuracy Adversarial": self.best_test_acc_adv})

    # federated average algorithm
    def federated_average_weights(self, local_weights):
        weights_avg = copy.deepcopy(local_weights[0])
        for key in weights_avg.keys():
            for i in range(1, len(local_weights)):
                weights_avg[key] += local_weights[i][key]
            weights_avg[key] = torch.div(weights_avg[key], len(local_weights))
        return weights_avg

    # evaluate performance on watermarking set
    def evaluate_watermark(self):
        self.global_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        device = 'cuda' if self.args.gpu else 'cpu'
        criterion = nn.CrossEntropyLoss().to(device)
        testloader = DataLoader(self.watermark_dataset, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = self.global_model(norm(images))
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        print("|---- Watermarking Test Accuracy: {:.2f}%".format(100*accuracy))
        wandb.log({"Watermarking Test Accuracy": 100*accuracy})
        return 100*accuracy

    # evaluate performance on test set
    def evaluate(self):
        # test_acc, test_loss = test_inference(args, final_global_model, test_dataset)
        self.global_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        device = 'cuda' if self.args.gpu else 'cpu'
        criterion = nn.CrossEntropyLoss().to(device)
        testloader = DataLoader(self.test_data, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = self.global_model(norm(images))
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
    
    # evaluates the adversarial performance
    def evaluate_adversarial(self):
        # test_acc, test_loss = test_inference(args, final_global_model, test_dataset)
        self.global_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        device = 'cuda' if self.args.gpu else 'cpu'
        criterion = nn.CrossEntropyLoss().to(device)
        # criterion = nn.NLLLoss().to(device)
        testloader = DataLoader(self.test_data, batch_size=128,
                                shuffle=False)
        
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            images = self.adversarial_attack(images, labels, self.global_model)
            # Inference
            outputs = self.global_model(norm(images))
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
        return 100*accuracy

    # generate adversarial samples
    def adversarial_attack(self, images, labels, model):
        #Vary the PGD attack parameters
        attacker = PGDAttack(model, self.args.pgd_eps, self.args.pgd_attack_steps, self.args.pgd_step_size)
        adv = attacker.perturb(images, labels)
        return adv