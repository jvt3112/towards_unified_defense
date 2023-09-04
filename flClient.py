"""
Some code snippets and functions used from: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/update.py
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from multiprocessing import cpu_count
import numpy as np
from pgd import PGDAttack
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
# from opacus.utils import convert_batchnorm_modules, replace_all_modules
from opacus.validators import ModuleValidator
import wandb
import copy

norm = transforms.Normalize([0.5], [0.5])
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class FLClient(object):
    def __init__(self, args, dataset, idxs, watermark_dataset=None, watermark_idxs=None):
        self.args = args
        # self.logger = logger
        self.watermarkloader = None
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        # self.watermark_trainloader, self.watermark_validloader, self.watermark_testloader = self.train_val_test(watermark_dataset, list(watermark_idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.train_sample_rate = args.dp_batch_size / len(dataset)

    def train_val_test(self, dataset, idxs, watermark_dataset=None, watermark_idxs=None):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(1*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                    batch_size=self.args.local_bs, shuffle=True)
        
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def train(self, model, global_round):
        # Set mode to train model
        if self.args.dp:
            if not ModuleValidator.is_valid(model):
              model = ModuleValidator.fix(model)
        model = model.to(self.device)
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=5e-4)
            if self.args.dp_relax:
                optimizer_relax = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=5e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        # using privacyengine of opacus library to incorporate differential privacy
        if self.args.dp:
            print('USING DP!')
            privacy_engine = PrivacyEngine()
            model, optimizer, self.trainloader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=self.trainloader,
                epochs=self.args.local_ep*self.args.epochs*self.args.frac,
                target_epsilon=self.args.dp_epsilon,
                target_delta=1e-3,
                max_grad_norm=1.0,
            )
        # training for n number of local epochs
        for iter in range(self.args.local_ep):
            batch_loss = []
            model.train()
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.args.adv:
                    images = self.mix_adv_samples(images, labels, model)
                optimizer.zero_grad()
                log_probs = model(norm(images))
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                
            if self.args.verbose and (batch_idx % 100 == 0) and iter%5==0:
                print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    global_round, iter, batch_idx * len(images),
                    len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()))
        
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            wandb.log({"Local epoch loss": sum(batch_loss)/len(batch_loss)})
            wandb.log({"Local epoch": iter+1})
        
        model.to(torch.device('cpu'))
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        return model.state_dict() , sum(epoch_loss) / len(epoch_loss)

    def test_inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(norm(images))
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

    def mix_adv_samples(self, images, labels, model):
        """ Mixes adversarial samples with clean samples.
            Replacing the 50% of the examples by Adversarial Examples. 
        """
        attacker = PGDAttack(model, self.args.pgd_eps, self.args.pgd_attack_steps, self.args.pgd_step_size)
        x_adv = attacker.perturb(images, labels)
        return x_adv