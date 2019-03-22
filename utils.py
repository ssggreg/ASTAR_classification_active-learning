import torch
from vogn import VOGN
use_cuda = torch.cuda.is_available()

def accuracy(model, dataloader, criterion=None):
    """ Computes the model's classification accuracy on the train dataset
        Computes classification accuracy and loss(optional) on the test dataset
        The model should return logits
    """
    model.eval()
    with torch.no_grad():
        correct = 0.
        running_loss = 0.
        for i, data in enumerate(dataloader):
            inputs, labels = data
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
        accuracy = correct / len(dataloader.dataset)
        if criterion is not None:
            running_loss = running_loss / len(dataloader)
            return accuracy, running_loss
    return accuracy

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    """
    Performs Training and Validation on test set on the given model using the specified optimizer
    :param model: (nn.Module) Model to be trained
    :param dataloaders: (list) train and test dataloaders
    :param criterion: Loss Function
    :param optimizer: Optimizer to be used for training
    :param num_epochs: Number of epochs to train the model
    :return: trained model, test and train metric history
    """
    train_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []
    test_loss_history = []
    trainloader, testloader = dataloaders

    for epoch in range(num_epochs):
        model.train(True)
        print('Epoch[%d]:' % epoch)
        running_train_loss = 0.
        for i, data in enumerate(trainloader):
            inputs, labels = data
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            if isinstance(optimizer, VOGN):
                def closure():
                    optimizer.zero_grad()
                    logits = model.forward(inputs)
                    loss = criterion(logits, labels)
                    return loss
            else:
                def closure():
                    optimizer.zero_grad()
                    logits = model.forward(inputs)
                    loss = criterion(logits, labels)
                    loss.backward()
                    return loss
            loss = optimizer.step(closure)
            running_train_loss += loss.detach().item()

            # Print Training Progress
            if i%200 == 199:
                train_accuracy = accuracy(model, trainloader)
                print('Iteration[%d]: Train Loss: %f   Train Accuracy: %f ' % (i+1, running_train_loss/i, train_accuracy))

        train_accuracy, train_loss = accuracy(model, trainloader, criterion)
        test_accuracy, test_loss = accuracy(model, testloader, criterion)
        train_accuracy_history.append(train_accuracy)
        train_loss_history.append(train_loss)
        test_accuracy_history.append(test_accuracy)
        test_loss_history.append(test_loss)
        print('## Epoch[%d], Train Loss: %f   &   Train Accuracy: %f' % (epoch, train_loss, train_accuracy))
        print('## Epoch[%d], Test Loss: %f   &   Test Accuracy: %f' % (epoch, test_loss, test_accuracy))
    return model, train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history

def inference(model, data, optimizer,mc_samples):
        
    inputs, labels = data

    if use_cuda:
        inputs, labels = inputs.cuda(), labels.cuda()
                
    predictions = optimizer.get_mc_predictions(model.forward,inputs,mc_samples=mc_samples)
    

    return predictions