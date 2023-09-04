import time
import os
import torch
import sys
import copy
import random
import numpy as np
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data import random_split, DataLoader, Dataset
from torch import nn
sys.path.append(os.path.abspath('../..'))
# from utils import get_dataset, average_weights, exp_details
# from flServer import FLServer
import argparse
import wandb
import warnings
warnings.filterwarnings("ignore")
from models import *
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from pgd import PGDAttack

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

def load_dataset(args):
    # initialize the train_dataset_watermark and user_groups_watermark 
    train_dataset_watermark = None 
    if args.dataset == 'cifar':
        torch.manual_seed(42)
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.CenterCrop(32),
            transforms.ToTensor()])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
    elif args.dataset == 'mnist' or 'fmnist':
        apply_transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor()])
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        else:
            data_dir = '../data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                          transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        if args.dataset == 'mnist':
            if args.wm:
                # Add watermark to training data
                load_watermark_dataset = torch.load('./watermarks/watermarks_samples_0.4_40.pt')
                train_dataset_watermark = DatasetSplit(load_watermark_dataset)
        if args.dataset == 'fmnist':
            if args.wm:
                load_watermark_dataset = torch.load('./watermarks/watermark_samples_fmnist_0.30_40_new.pt')
                train_dataset_watermark = DatasetSplit(load_watermark_dataset)
    return train_dataset, test_dataset, train_dataset_watermark

def mix_adv_samples(args, images, labels, model, pgd_eps, pgd_attack_steps, pgd_step_size, frac=1):
    # Mix adversarial samples with clean samples (only 50% of the samples are adversarial)
    rand_perm = np.random.permutation(images.size(0))
    rand_perm = rand_perm[:int(frac*rand_perm.size)]
    x_adv, y_adv = images[rand_perm, :], labels[rand_perm]
    attacker = PGDAttack(model, pgd_eps, pgd_attack_steps, pgd_step_size)
    x_adv = attacker.perturb(x_adv, y_adv)
    images[rand_perm,:] = x_adv
    return images

def evaluate_watermark(args, model, training_dataset_watermark):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(training_dataset_watermark, batch_size=128,
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
    print("|---- Watermarking Test Accuracy: {:.2f}%".format(100*accuracy))
    wandb.log({"Watermarking Test Accuracy": 100*accuracy})
    return 100*accuracy

def evaluate(args, model, test_dataset):
    # test_acc, test_loss = test_inference(args, final_model, test_dataset)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
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
    wandb.log({"Test Loss": loss})
    return 100*accuracy, loss

def membership_check_optimized(args, model_target, train_dataset, test_dataset):
    model_target.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Helper function to compute confidence values for a dataset
    def compute_confidence(dataset):
        confidence = []
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_target(norm(images))
            _, pred_labels = torch.max(outputs, 1)
            confidence.extend(F.softmax(outputs, dim=1).max(dim=1)[0].detach().cpu().numpy())
        return np.array(confidence)

    # Compute confidence values for train and test datasets
    attack_model_dataset_full_confidence_train = compute_confidence(train_dataset)
    attack_model_dataset_full_confidence_test = compute_confidence(test_dataset)

    # Compute ratios and accuracy for different confidence thresholds
    sort_confidence = np.sort(np.concatenate((attack_model_dataset_full_confidence_train,
                                              attack_model_dataset_full_confidence_test)))
    max_accuracy = 0.5
    best_precision = 0.5
    best_recall = 0.5
    for delta in sort_confidence:
        ratio1 = np.sum(attack_model_dataset_full_confidence_train >= delta) / len(attack_model_dataset_full_confidence_train)
        ratio2 = np.sum(attack_model_dataset_full_confidence_test >= delta) / len(attack_model_dataset_full_confidence_test)
        accuracy_now = 0.5 * (ratio1 + 1 - ratio2)
        if accuracy_now > max_accuracy:
            max_accuracy = accuracy_now
            best_precision = ratio1 / (ratio1 + ratio2)
            best_recall = ratio1

    wandb.log({"Membership Inference Accuracy": 100 * max_accuracy})
    wandb.log({"Precision": 100 * best_precision})
    wandb.log({"Recall": 100 * best_recall})

def membership_check_optimized_adv(args, model_target, train_dataset, test_dataset, pgd_eps, pgd_attack_steps, pgd_step_size, frac=1):
    model_target.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Helper function to compute confidence values for a dataset
    def compute_confidence(dataset):
        confidence = []
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            images = mix_adv_samples(args, images, labels, model_target, pgd_eps, pgd_attack_steps, pgd_step_size, frac=frac)
            outputs = model_target(norm(images))
            _, pred_labels = torch.max(outputs, 1)
            confidence.extend(F.softmax(outputs, dim=1).max(dim=1)[0].detach().cpu().numpy())
        return np.array(confidence)

    # Compute confidence values for train and test datasets
    attack_model_dataset_full_confidence_train = compute_confidence(train_dataset)
    attack_model_dataset_full_confidence_test = compute_confidence(test_dataset)

    # Compute ratios and accuracy for different confidence thresholds
    sort_confidence = np.sort(np.concatenate((attack_model_dataset_full_confidence_train,
                                              attack_model_dataset_full_confidence_test)))
    max_accuracy = 0.5
    best_precision = 0.5
    best_recall = 0.5
    for delta in sort_confidence:
        ratio1 = np.sum(attack_model_dataset_full_confidence_train >= delta) / len(attack_model_dataset_full_confidence_train)
        ratio2 = np.sum(attack_model_dataset_full_confidence_test >= delta) / len(attack_model_dataset_full_confidence_test)
        accuracy_now = 0.5 * (ratio1 + 1 - ratio2)
        if accuracy_now > max_accuracy:
            max_accuracy = accuracy_now
            best_precision = ratio1 / (ratio1 + ratio2)
            best_recall = ratio1

    wandb.log({"Membership Adv Inference Accuracy": 100 * max_accuracy})
    wandb.log({"Precision Adv": 100 * best_precision})
    wandb.log({"Recall Adv": 100 * best_recall})

def evaluate_train(args, model, test_dataset):
    # test_acc, test_loss = test_inference(args, final_model, test_dataset)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
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
    print("|---- Training Accuracy: {:.2f}%".format(100*accuracy))
    wandb.log({"Training Accuracy": 100*accuracy})
    return 100*accuracy

def evaluate_adversarial(args, model, test_dataset, pgd_eps, pgd_attack_steps, pgd_step_size, frac=1):
    # test_acc, test_loss = test_inference(args, final_model, test_dataset)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)
    
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        images = mix_adv_samples(args, images, labels, model, pgd_eps, pgd_attack_steps, pgd_step_size, frac=frac)
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
    wandb.log({"PGD Epsilon": pgd_eps})
    wandb.log({"PGD Attack Steps": pgd_attack_steps})
    return 100*accuracy


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    # model arguments
    parser.add_argument('--model', type=str, default='lenet', help='model name')
   
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")
    parser.add_argument('--gpu', type=int, default=1, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='random seed')
    # mechanisms arguments 

    # DP arguments
    parser.add_argument('--dp', type=int, default=0, help='whether to use dp')
    parser.add_argument('--dp_epsilon', type=float, default=5.0,
                        help="the epsilon for dp model")
    parser.add_argument('--dp_relax', type=int, default=0,
                        help="whether to use relaxed dp")
    parser.add_argument('--grad_norm', type=float, default=1.0,
                        help="maximum gradient norm")

    # Adversarial training arguments (PGD)
    parser.add_argument('--adv', type=int, default=0, help='whether to use adversarial training')
    parser.add_argument('--pgd_eps', type=float, default=0.25, help='epsilon for PGD')
    parser.add_argument('--pgd_attack_steps', type=int, default=25, help='number of steps for PGD')
    parser.add_argument('--pgd_step_size', type=float, default=0.01, help='step size for PGD')
    # relaxed adversarial training arguments
    parser.add_argument('--adv_relax', type=int, default=0, help='whether to use relaxed adversarial training')
    parser.add_argument('--adv_percent', type=float, default=1, help='percentage of adversarial examples in a batch')
    # Watermarking arguments
    parser.add_argument('--wm', type=int, default=0, help='whether to use watermarking')
    parser.add_argument('--save', type=int, default=0, help='saving a model or not')
    parser.add_argument('--num', type=int, default=1, help='neighbouring')

    parser.add_argument('--mem', type=int, default=0, help='memebrship')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    args = args_parser()
    wandb.init(
        project="final-all-centralised",
        name= str(args.dataset)+'_'+str(args.model)+'_'+str(args.lr)+'_'+str(args.dp)+'_'+str(args.adv)+'_'+str(args.wm),
        config={
            "epochs": args.epochs,
            "lr": args.lr,
            "dp_epsilon": args.dp_epsilon,
            "dataset": args.dataset,
            "pgd_epsilon": args.pgd_eps,
            "pgd_attack_steps": args.pgd_attack_steps,
            "pgd_step_size": args.pgd_step_size,
            "dp_relax": args.dp_relax,
            "dp": args.dp,
            "adv_percent": args.adv_percent,
            "adv_relax": args.adv_relax,
            "adv": args.adv,
            "wm": args.wm,
            "model": args.model
        }
    )

    if args.gpu:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'
    training_dataset1, test_dataset , training_dataset_watermark = load_dataset(args)
    training_dataset = DataLoader(training_dataset1, batch_size=128, shuffle=True)
    # test_dataset = DataLoader(test_dataset, batch_size=128, shuffle=False)  
    if args.wm:
        training_dataset_watermark_loader = DataLoader(training_dataset_watermark, batch_size=16, shuffle=True)
   
    norm = transforms.Normalize([0.5], [0.5])
    print('Norm:', norm)

    if args.model=='small-cnn':
        model = MNIST_CNN_SMALL()
        global_model = MNIST_CNN_SMALL()
    elif args.model=='cnn':
        model = MNIST_CNN()
        global_model = MNIST_CNN()
    elif args.model=='lenet':
        if args.dataset == 'cifar':
            model = MNIST_CNN_SMALL_LENET(3)
            global_model = MNIST_CNN_SMALL_LENET(3)
        else:
            model = MNIST_CNN_SMALL_LENET(1)
            global_model = MNIST_CNN_SMALL_LENET(1)
    elif args.model=='resnet':
        model = resnet18(10)
        global_model = resnet18(10)
    model.to(device)
    global_model.to(device)
    if args.dp:
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
        if not ModuleValidator.is_valid(global_model):
            global_model = ModuleValidator.fix(global_model)

    print(model)
    model.train()
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
    if args.dp_relax:
        optimizer_relax = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
    if args.dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, training_dataset = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=training_dataset,
            epochs= args.epochs,
            target_epsilon=args.dp_epsilon,
            target_delta=1e-3,
            max_grad_norm=args.grad_norm,
        )
    epoch_loss = [] 
    best_test_acc = 0
    best_test_epoch = 0
    best_test_acc_watermark = 0
    best_test_acc_adv = 0
    print('Training')
    for iter in range(args.epochs):
        batch_loss = []
        model.train()
        for batch_idx, (images, labels) in enumerate(training_dataset):
            images, labels = images.to(device), labels.to(device)
            if args.adv:
                images = mix_adv_samples(args, images, labels, model,  args.pgd_eps, args.pgd_attack_steps, args.pgd_step_size, 1)
            optimizer.zero_grad()
            log_probs = model(norm(images))
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            
        if args.verbose and (batch_idx % 100 == 0) and iter%5==0:
            print('| Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                iter, batch_idx * len(images),
                len(training_dataset.dataset),
                100. * batch_idx / len(training_dataset), loss.item()))
    
        if args.wm:
            watermark_batch_loss = []
            for batch_idx, (images, labels) in enumerate(training_dataset_watermark_loader):
                images, labels = images.to(device), labels.to(device)
                if args.dp:
                    model.zero_grad()
                    log_probs = model(norm(images))
                    loss_wm = criterion(log_probs, labels)
                    loss_wm.backward()
                    optimizer_relax.step()
                else:
                    optimizer.zero_grad()
                    log_probs = model(norm(images))
                    loss_wm = criterion(log_probs, labels)
                    loss_wm.backward()
                    optimizer.step()
                watermark_batch_loss.append(loss_wm.item())
        if args.wm:
            epoch_loss.append(((sum(batch_loss)/len(batch_loss))+(sum(watermark_batch_loss)/len(watermark_batch_loss)))/2)
        else:
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        wandb.log({"Epoch loss": epoch_loss[-1]})
        wandb.log({"Epoch": iter+1})
        trainAccu = evaluate_train(args, model, training_dataset1)
        testAccu, test_loss = evaluate(args, model, test_dataset)
        
        if args.mem:
            membership_check_optimized(args, model, training_dataset1, test_dataset)
        # wandb.log({"Test Accuracy": testAccu})
        global_model.load_state_dict({k.replace("_module.", ""): v for k, v in model.state_dict().items()})
        if args.wm:
            watermarkAcc = evaluate_watermark(args, model, training_dataset_watermark)
        if args.adv:
            advAccu = evaluate_adversarial(args, global_model, test_dataset, args.pgd_eps, args.pgd_attack_steps, args.pgd_step_size)
        if args.wm and args.adv:
            if (testAccu+advAccu+watermarkAcc) > (best_test_acc+best_test_acc_watermark+best_test_acc_adv):
                best_test_acc = testAccu
                best_test_epoch = iter
                best_test_acc_watermark = watermarkAcc
                best_test_acc_adv = advAccu
        elif args.wm:
            if (testAccu+watermarkAcc) > (best_test_acc+best_test_acc_watermark):
                best_test_acc = testAccu
                best_test_epoch = iter
                best_test_acc_watermark = watermarkAcc
        elif args.adv:
            if (testAccu+advAccu) > (best_test_acc+best_test_acc_adv):
                best_test_acc = testAccu
                best_test_epoch = iter
                best_test_acc_adv = advAccu
        else:
            if testAccu > best_test_acc:
                best_test_acc = testAccu
                best_test_epoch = iter
            # torch.save(model.state_dict(), 'watermarked_0.5.pt')
        # membership_check_optimized_adv(args, global_model, training_dataset1, test_dataset, args.pgd_eps, args.pgd_attack_steps, args.pgd_step_size)
    wandb.log({"Best Test Epoch": best_test_epoch})
    wandb.log({"Best Test Accuracy": best_test_acc})
    if args.wm:
        wandb.log({"Best Test Accuracy Watermark": best_test_acc_watermark})
    if args.adv:
        wandb.log({"Best Test Accuracy Adversarial": best_test_acc_adv})

    global_model.load_state_dict({k.replace("_module.", ""): v for k, v in model.state_dict().items()})
    if args.adv:
        evaluate_adversarial(args, global_model, test_dataset, args.pgd_eps, args.pgd_attack_steps, args.pgd_step_size)
    if args.wm:
        evaluate_watermark(args, global_model, training_dataset_watermark)
    if args.save==1:
        torch.save(global_model.state_dict(), 'lenet_no_defense_CIFAR_final.pt')
    if args.save==2:
        torch.save(global_model.state_dict(), 'lenet_only_dp_CIFAR_final.pt')
    if args.save==3:
        torch.save(global_model.state_dict(), 'lenet_adv_dp_CIFAR.pt')
    if args.save==4:
        torch.save(global_model.state_dict(), 'lenet_only_adv_CIFAR_final.pt')
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    wandb.log({"Total Run Time": time.time()-start_time})
    wandb.finish()
