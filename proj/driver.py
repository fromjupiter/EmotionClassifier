import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from PCA import MyPCA
from SoftmaxRegression import SoftmaxRegression
from kfold import MyKFold
from dataloader import load_data, balanced_sampler

parser = argparse.ArgumentParser(description='Singer Match Entry Point')

parser.add_argument('-r', '--routine', default='test', help='Run predefined routine')
parser.add_argument('-d', '--dir', default='./aligned/', help='data directory')
parser.add_argument('-e', '--emotions', default='happiness, anger', help='delimited emotions', type=str)
args = parser.parse_args()
emotions = [x.strip() for x in args.emotions.split(',')]
data_dir = args.dir

class TrainResult(object):
    def __init__(self):
        self.best_coef = None
        self.train_losses = []
        self.train_accuracies = []
        self.valid_losses = []
        self.valid_accuracies = []
        self.test_loss = None
        self.test_accuracy = None

class AggregateResult(object):
    def __init__(self):
        self.train_loss_avg = None
        self.train_loss_std = None
        self.valid_loss_avg = None
        self.valid_loss_std = None
        self.train_accuracy_avg = None
        self.train_accuracy_std = None
        self.valid_accuracy_avg = None
        self.valid_accuracy_std = None

    def aggregate(res_list):
        if res_list is None or len(res_list)==0: return AggregateResult()
        res = AggregateResult()
        res.train_loss_avg = np.average(list(map(lambda x:x.train_losses, res_list)), axis=0)
        res.train_loss_std = np.std(list(map(lambda x:x.train_losses, res_list)), axis=0)
        res.valid_loss_avg = np.average(list(map(lambda x:x.valid_losses, res_list)), axis=0)
        res.valid_loss_std = np.std(list(map(lambda x:x.valid_losses, res_list)), axis=0)

        res.train_accuracy_avg = np.average(list(map(lambda x:x.train_accuracies, res_list)), axis=0)
        res.train_accuracy_std = np.std(list(map(lambda x:x.train_accuracies, res_list)), axis=0)
        res.valid_accuracy_avg = np.average(list(map(lambda x:x.valid_accuracies, res_list)), axis=0)
        res.valid_accuracy_std = np.std(list(map(lambda x:x.valid_accuracies, res_list)), axis=0)
        return res

# return TrainResult
def trainModel(model, d_train, d_valid, d_test, epoch=100, n_components=None, useBatch=True):
    X_train, y_train = d_train
    X_valid, y_valid = d_valid
    X_test, y_test = d_test
    if n_components is not None:
        pca = MyPCA(n_components=n_components)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_valid = pca.transform(X_valid)
        X_test = pca.transform(X_test)
    
    #add bias term
    X_train = np.concatenate((np.ones((X_train.shape[0],1)) , X_train), axis=1)
    X_valid = np.concatenate((np.ones((X_valid.shape[0],1)) , X_valid), axis=1)
    X_test = np.concatenate((np.ones((X_test.shape[0],1)) , X_test), axis=1)

    res = TrainResult()

    best_loss = float('inf')
    
    # add init status
    model.setup(X_train, y_train)
    res.train_losses.append(model._loss(X_train, y_train))
    res.valid_losses.append(model._loss(X_valid, y_valid))
    res.train_accuracies.append(sum(model.predict(X_train)==y_train)/len(y_train))
    res.valid_accuracies.append(sum(model.predict(X_valid)==y_valid)/len(y_valid))
    for i in range(epoch):
        model.fit_one_epoch(X_train, y_train, useBatch)
        valid_loss = model._loss(X_valid, y_valid)
        if valid_loss < best_loss:
            best_loss = valid_loss
            res.best_coef = model.coef_
        res.train_losses.append(model._loss(X_train, y_train))
        res.valid_losses.append(valid_loss)
        res.train_accuracies.append(sum(model.predict(X_train)==y_train)/len(y_train))
        res.valid_accuracies.append(sum(model.predict(X_valid)==y_valid)/len(y_valid))

    res.test_loss = model._loss(X_test, y_test)
    res.test_accuracy = sum(model.predict(X_test)==y_test)/len(y_test)
    res.train_accuracies = np.array(res.train_accuracies)
    res.train_losses = np.array(res.train_losses)
    res.valid_accuracies = np.array(res.valid_accuracies)
    res.valid_losses = np.array(res.valid_losses)
    
    model.coef_ = res.best_coef
    return res

#############################
# All routines start below!!
#############################

def test():
    pass



############################
#
# Visualize Softmax Regression weights
#
############################

def visualizeWeights():
    # hyper parameters
    EPOCH = 100
    N_COMPONENTS = 50
    N_SPLITS = 10
    LEARNING_RATE = 0.01

    kfold = MyKFold(n_splits=N_SPLITS)
    test_accuracy = 0
    test_loss = 0
    
    classifier = SoftmaxRegression(lr=LEARNING_RATE)
    for d_train, d_valid, d_test in kfold.split_dict(images):
        result = trainModel(classifier, d_train, d_valid, d_test, epoch=EPOCH, n_components=N_COMPONENTS)
        break
    print("test loss: {}, test accuracy: {}.".format(result.test_loss, result.test_accuracy))
    


############################
#
# Report Softmax Regression Training process
#
############################
def softmaxSGD():
    EPOCH = 50
    LEARNING_RATE = 0.1
    gd_res = doSoftmax(EPOCH, LEARNING_RATE, useBatch=True)
    sgd_res = doSoftmax(EPOCH, LEARNING_RATE, useBatch=False)
    xlabels = [x for x in range(0,EPOCH+1)]
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.suptitle('Softmax Regression GD vs SGD')
    axs[0].set_title('Cross-Entropy Loss')
    axs[1].set_title('Accuracy(Percent Correct)')
    axs[1].set_xlabel('Epoch')

    axs[0].errorbar(xlabels,gd_res.train_loss_avg,gd_res.train_loss_std, errorevery=10, label='GD')
    axs[0].errorbar(xlabels,sgd_res.train_loss_avg,sgd_res.train_loss_std, errorevery=10, label='SGD')
    axs[1].errorbar(xlabels,gd_res.train_accuracy_avg,gd_res.train_accuracy_std, errorevery=10, label='GD')
    axs[1].errorbar(xlabels,sgd_res.train_accuracy_avg,sgd_res.train_accuracy_std, errorevery=10, label='SGD')
    axs[0].legend()
    axs[1].legend()
    plt.show()

def reportSoftmax():
    EPOCH = 100
    LEARNING_RATE = 0.5
    result = doSoftmax(EPOCH, LEARNING_RATE, useBatch=True)
    
    xlabels = [x for x in range(0,EPOCH+1)]
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.suptitle('Softmax Regression GD Process')
    axs[0].set_title('Cross-Entropy Loss')
    axs[1].set_title('Accuracy(Percent Correct)')
    axs[1].set_xlabel('Epoch')

    axs[0].errorbar(xlabels,result.train_loss_avg,result.train_loss_std, errorevery=10, label='train set')
    axs[0].errorbar(xlabels,result.valid_loss_avg,result.valid_loss_std, errorevery=10, label='holdout set')
    axs[1].errorbar(xlabels,result.train_accuracy_avg,result.train_accuracy_std, errorevery=10, label='train set')
    axs[1].errorbar(xlabels,result.valid_accuracy_avg,result.valid_accuracy_std, errorevery=10, label='holdout set')
    axs[0].legend()
    axs[1].legend()
    plt.show()

def doSoftmax(epoch=100, lr=0.1, useBatch=True):
    # hyper parameters
    N_COMPONENTS = 50
    N_SPLITS = 10

    kfold = MyKFold(n_splits=N_SPLITS)
    results = []
    test_accuracy = 0
    test_loss = 0
    for d_train, d_valid, d_test in kfold.split_dict(images):
        classifier = SoftmaxRegression(lr=lr)
        result = trainModel(classifier, d_train, d_valid, d_test, epoch=epoch, n_components=N_COMPONENTS, useBatch=useBatch)
        test_loss += result.test_loss
        test_accuracy += result.test_accuracy
        results.append(result)

    test_accuracy /= N_SPLITS
    test_loss /= N_SPLITS
    print("useBatch={}, test loss: {}, test accuracy: {}.".format(useBatch, test_loss, test_accuracy))
    return AggregateResult.aggregate(results)


############################
#
# Report balanced Softmax Regression on imbalanced dataset
#
############################

def reportBalancedSoftmax():
    images, cnt = load_data(data_dir)
    # hyper parameters
    EPOCH = 50
    N_COMPONENTS = 50
    N_SPLITS = 10
    LEARNING_RATE = 0.01

    kfold = MyKFold(n_splits=N_SPLITS)
    results = []
    test_accuracy = 0
    test_loss = 0
    for d_train, d_valid, d_test in kfold.split_dict(images):
        classifier = SoftmaxRegression(lr=LEARNING_RATE)
        result = trainModel(classifier, d_train, d_valid, d_test, epoch=EPOCH, n_components=N_COMPONENTS)
        test_loss += result.test_loss
        test_accuracy += result.test_accuracy
        results.append(result)

    test_accuracy /= N_SPLITS
    test_loss /= N_SPLITS 
    print("test loss: {}, test accuracy: {}.".format(test_loss, test_accuracy))
    print(classifier.classes_)
    print(classifier.weights_)


############################
#
# Report PCA
#
############################

def reportPCA():
    # hyper parameters
    EPOCH = 50
    N_SPLITS = 10
    LEARNING_RATE = 0.01

    if len(emotions)<=2:
        n_components = [3, 5, 8, 10]
    else:
        n_components = [10, 20, 40, 50]
    
    kfold = MyKFold(n_splits=N_SPLITS)
    df = pd.DataFrame(index=np.arange(0, len(n_components)), columns=['n_components', 'avg_accuracy', 'avg_loss'])
    for i, n in enumerate(n_components):
        accuracy = 0
        loss = 0
        for d_train, d_valid, d_test in kfold.split_dict(images):
            classifier = SoftmaxRegression(lr=LEARNING_RATE)
            result = trainModel(classifier, d_train, d_valid, d_test, epoch=EPOCH, n_components=n)
            accuracy += result.test_accuracy
            loss += result.test_loss
        accuracy /= kfold.n_splits
        loss /= kfold.n_splits
        df.loc[i] = (n, accuracy, loss)
    print("PCA Result (reported loss/accuracy on test set):")
    print(df)


dataset, cnt = load_data(data_dir)
images = balanced_sampler(dataset, cnt, emotions=emotions)

print('--- Below is the routine output ---')
locals()[args.routine]()