import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from data_loader import ToyDataset
from model.model import GraphPairNN

def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

# Early stopping class
class EarlySopping():
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.model = None
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = -val_loss
            self.checkpoint(val_loss, model)
        elif -val_loss < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = -val_loss
            self.checkpoint(val_loss, model)
            self.counter = 0

    def checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def train(model, optimizer, scheduler, loss_fn, x_data, y_data):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        scheduler: (torch.optim.lr_scheduler) scheduler to adapt learning rate of optimizer
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        x_data: input training data
        y_data: output training data
    """

    # set model to training mode
    model.train()

    # clear previous gradients, compute gradients of all variables wrt loss
    optimizer.zero_grad()

    # reshape data for loss function

    logits = model(x_data)
    logits = logits.reshape(shape=(x_data.shape[0]*x_data.shape[1], x_data.shape[1]))
    target = y_data.reshape(shape=(y_data.shape[0]*y_data.shape[1],))
    loss = loss_fn(logits, target)
    loss.backward()

    # performs updates using calculated gradients
    optimizer.step()
    scheduler.step()

    return loss.item()


def evaluate(model, criterion, x_data, y_data):
    model.eval()

    logits = model(x_data)
    logits = logits.reshape(shape=(x_data.shape[0] * x_data.shape[1], x_data.shape[1]))
    target = y_data.reshape(shape=(y_data.shape[0] * y_data.shape[1],))
    loss = criterion(logits, target)
    preds = torch.max(logits, dim=1)[1]
    correct = preds.eq(target).double()
    accuracy = correct.sum() / logits.shape[0]

    return loss.item(), preds, accuracy.item()


def train_and_evaluate(model, data, optimizer, scheduler, loss_fn, params, dev):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        data: (tuple of ndarrays) x and y data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        scheduler: scheduler to adapt learning rate during training
        params: (Params) hyperparameters
        dev: the device to which to copy tensor objects
    """

    # Splitting data into train, val, test
    x_train, x_validate, x_test = np.split(data[0], [int(.6*len(data[0])), int(.8*len(data[0]))])
    y_train, y_validate, y_test = np.split(data[1], [int(.6 * len(data[1])), int(.8 * len(data[1]))])

    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    x_validate = torch.from_numpy(x_validate).type(torch.FloatTensor)
    y_validate = torch.from_numpy(y_validate).type(torch.LongTensor)
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    # Copying the data to the correct device
    x_train.to(dev)
    x_validate.to(dev)
    x_test.to(dev)
    y_train.to(dev)
    y_validate.to(dev)
    y_test.to(dev)

    # Early stopping setup
    stop = EarlySopping(params['patience'], params['delta'])

    best = 0.0
    results = []

    for epoch in range(0, opt['epochs']):

        # compute number of batches in one epoch (one full pass over the training set)
        # -----------------------
        # Train Model
        # -----------------------
        loss = train(model, optimizer, scheduler, loss_fn, x_train, y_train)
        # -----------------------
        # Evaluate Model
        # -----------------------
        _, preds, accuracy_dev = evaluate(model, loss_fn, x_validate, y_validate)
        # -----------------------
        # Test Model
        # -----------------------
        _, preds, accuracy_test = evaluate(model, loss_fn, x_test, y_test)

        print(
            'Epoch: {} | Loss: {:.3f} | Dev acc: {:.3f} | Test acc: {:.3f}'.format(
                epoch,
                loss,
                accuracy_dev,
                accuracy_test))
        results += [(accuracy_dev, accuracy_test)]
        if accuracy_dev > best:
            best = accuracy_dev
            state = dict([('model', copy.deepcopy(model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])
        # -----------------------
        # Early stopping
        # -----------------------
        # stop(accuracy_dev, model)
        # if stop.early_stop:
        #     print("Early stopping")
        #     break

    _, preds, accuracy_test = evaluate(model, loss_fn, x_test, y_test)
    print(str(accuracy_test))

        



def main(params, dev):

    # Load data (for now toy data)
    data = ToyDataset(1,size=1024,n_pools=10,samples=10,input_size=32,dist='perm')

    # Creating model
    model = GraphPairNN(params['input_size'], params['input_size'], params['input_size'])
    model_param = [p for p in model.parameters() if p.requires_grad]

    # Creating optimizer and scheduler
    optimizer = get_optimizer(params['optimizer'], model_param, params['lr'], params['decay'])
    scheduler = StepLR(optimizer, params['step'], params['gamma'])

    train_and_evaluate(model, (data.get_x(), data.get_y()), optimizer, scheduler, nn.CrossEntropyLoss(), params, dev)






    return None

if __name__ == '__main__':
    #-------------------------------
    # Parsing program input
    #-------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--save', type=str, default='/')
    parser.add_argument('--input_size', type=int, default=32, help='Size of input to neural net.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs per iteration.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--step', type=int, default=50,
                        help='Step size on epochs after which learning rate is changed.')
    parser.add_argument('--gamma', type=float, default=1.0, help='Multiplicative factor applied to the learning after '
                                                                 'step many epochs.')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping.')
    parser.add_argument('--delta', type=float, default=0.001, help='Delta for early stopping.')
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA, else cuda is used if available.')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device('cuda' if use_cuda else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    opt = vars(args)

    main(opt, device)
