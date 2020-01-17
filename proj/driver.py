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

# return average and std of train/validation loss
def mergeResults(res_list):
    if res_list is None or len(res_list)==0: return None

    train_avg = np.average(list(map(lambda x:x.train_losses, res_list)), axis=0)
    train_std = np.std(list(map(lambda x:x.valid_losses, res_list)), axis=0)
    valid_avg = np.average(list(map(lambda x:x.valid_losses, res_list)), axis=0)
    valid_std = np.std(list(map(lambda x:x.valid_losses, res_list)), axis=0)
    return (train_avg, train_std, valid_avg, valid_std)

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
    # hyper parameters
    N_COMPONENTS = 40
    N_SPLITS = 10
    LEARNING_RATE = 0.01

    kfold = MyKFold(n_splits=N_SPLITS)

    i=-1
    for d_train, d_valid, d_test in kfold.split_dict(images):
        i+=1
        # test only 9th split
        if i!=9: 
            continue
        
        classifier = SoftmaxRegression(lr=LEARNING_RATE)
        result = trainModel(classifier, d_train, d_valid, d_test, epoch=50, n_components=5)
        print(result.test_loss)
        print(result.test_accuracy)
        print(result.train_losses)
        print(result.valid_losses)



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
def reportSoftmax():
    EPOCH = 51
    train_avg, train_std, valid_avg, valid_std = doSoftmax(EPOCH, useBatch=True)
    train_avg_sgd, train_std_sgd, valid_avg_sgd, valid_std_sgd = doSoftmax(EPOCH, useBatch=False)
    
    xlabels = [x for x in range(0,EPOCH)] # np.array([1] + [x for x in range(10, EPOCH+1, 10)])
    fig, axs = plt.subplots()
    axs.set_title('Softmax Regression Training Process')
    axs.set_ylabel('Error(Average Cross-Entropy)')
    axs.set_xlabel('GD epoch')
    axs.errorbar(xlabels,train_avg,train_std, errorevery=10, label='gd training loss')
    axs.errorbar(xlabels,valid_avg,valid_std, errorevery=10, label='gd holdout loss')
    axs.errorbar(xlabels,train_avg_sgd,valid_std_sgd, errorevery=10, label='sgd training loss')
    axs.errorbar(xlabels,valid_avg_sgd,valid_std_sgd, errorevery=10, label='sgd holdout loss')
    axs.legend()
    axs.legend()
    plt.show()
    # fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    # fig.suptitle('Softmax Regression Training Process')
    # axs[0].set_ylabel('Error(Average Cross-Entropy)')
    # axs[0].set_xlabel('GD epoch')
    # axs[0].set_title('GD')
    # axs[0].errorbar(xlabels,train_avg,train_std, errorevery=10, label='training loss')
    # axs[0].errorbar(xlabels,valid_avg,valid_std, errorevery=10, label='holdout loss')
    # axs[1].set_title('SGD')
    # axs[1].errorbar(xlabels,train_avg_sgd,valid_std_sgd, errorevery=10, label='training loss')
    # axs[1].errorbar(xlabels,valid_avg_sgd,valid_std_sgd, errorevery=10, label='holdout loss')
    # axs[0].legend()
    # axs[1].legend()
    plt.show()

def doSoftmax(epoch=100, useBatch=True):
    # hyper parameters
    N_COMPONENTS = 50
    N_SPLITS = 10
    LEARNING_RATE = 0.01

    kfold = MyKFold(n_splits=N_SPLITS)
    results = []
    test_accuracy = 0
    test_loss = 0
    for d_train, d_valid, d_test in kfold.split_dict(images):
        classifier = SoftmaxRegression(lr=LEARNING_RATE)
        result = trainModel(classifier, d_train, d_valid, d_test, epoch=epoch, n_components=N_COMPONENTS, useBatch=useBatch)
        test_loss += result.test_loss
        test_accuracy += result.test_accuracy
        results.append(result)

    test_accuracy /= N_SPLITS
    test_loss /= N_SPLITS
    print("useBatch={}, test loss: {}, test accuracy: {}.".format(useBatch, test_loss, test_accuracy))
    return mergeResults(results)


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