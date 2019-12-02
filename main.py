import argparse
import torch
import torchvision
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from model import BaseModel
from train import train, resume, evaluate
import numpy as np
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./Result')
show=ToPILImage()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='df_0')

    parser.add_argument('--resume', type=int, default=1, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    epoch=args.epochs
    # dataloaders
    transform = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainvalset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transform)
    trainset, testset = torch.utils.data.random_split(trainvalset, [49000, 1000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=2)
    dataloaders = (trainloader, testloader)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




    # network
    model = BaseModel(args).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))

    # resume the trained model
    if args.resume:
        model, optimizer = resume(model, optimizer)

    if args.test == 1: # test mode, resume the trained model and test
        testing_accuracy = evaluate(model, testloader)
        print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))

    else: # train mode
        train(args, model, optimizer, dataloaders)
        print('training finished')
