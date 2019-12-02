import os
import time
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from tensorboardX import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
show=ToPILImage()
criterion = torch.nn.CrossEntropyLoss()
classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def train(args, model, optimizer, dataloaders):
    trainloader, testloader = dataloaders

    best_testing_accuracy = 0.0
    train_loss=[]
    epoch_x=[]
    # training
    for epoch in range(args.epochs):
        model.train()

        batch_time = time.time(); iter_time = time.time()
        for i, data in enumerate(trainloader):

            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)

            cls_scores = model(imgs)
            loss = criterion(cls_scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0 and i != 0:
                print('epoch:{}, iter:{}, time:{:.2f}, loss:{:.5f}'.format(epoch, i,
                    time.time()-iter_time, loss.item()))
                iter_time = time.time()
        batch_time = time.time() - batch_time
        train_loss.append(loss.item())
        epoch_x.append(epoch)

        print('[epoch {} | time:{:.2f} | loss:{:.5f}]'.format(epoch, batch_time, loss.item()))
        print('-------------------------------------------------')

        if epoch % 1 == 0:
            testing_accuracy = evaluate(model, testloader)
            print('testing accuracy: {:.3f}'.format(testing_accuracy))

            if testing_accuracy > best_testing_accuracy:
                ### compare the previous best testing accuracy and the new testing accuracy
                ### save the model and the optimizer --------------------------------
                print('Saving....')
                state = {
                    'state_dict': model.state_dict(),
                    'acc': testing_accuracy,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,}
                torch.save(state,'model_checkpoint.pth')
                best_testing_accuracy=testing_accuracy

                ## YOUR CODE HERE


                ### -----------------------------------------------------------------
                print('new best model saved at epoch: {}'.format(epoch))
    plt.plot(epoch_x, train_loss)
    plt.show()

def evaluate(model, testloader):
    total_count = torch.tensor([0.0]); correct_count = torch.tensor([0.0])
    for i, data in enumerate(testloader):
        imgs, labels = data
        show(torchvision.utils.make_grid(imgs/2-0.5)).resize((400,100))

        imgs, labels = imgs.to(device), labels.to(device)
        total_count += labels.size(0)

        with torch.no_grad():
            cls_scores = model(imgs)

            predict = torch.argmax(cls_scores, dim=1)

            #pre = predict.numpy()
            correct_count += (predict == labels).sum().type(torch.FloatTensor)
    testing_accuracy = correct_count / total_count
    return testing_accuracy.item()


def resume(model, optimizer):
    checkpoint_path = './model_checkpoint.pth'
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    ### load the model and the optimizer --------------------------------
    model_CKPT = torch.load(checkpoint_path)
    model.load_state_dict(model_CKPT['state_dict'])
    optimizer.load_state_dict(model_CKPT['optimizer'])
    ## YOUR CODE HERE


    ### -----------------------------------------------------------------

    print('Resume completed for the model\n')

    return model, optimizer
