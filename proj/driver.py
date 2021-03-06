import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score

from PCA import MyPCA
from SoftmaxRegression import SoftmaxRegression
from LogisticRegression import LogisticRegression
from kfold import MyKFold
from dataloader import load_data, balanced_sampler, display_face

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
        self.train_bers = []
        self.valid_bers = []
        self.test_loss = None
        self.test_accuracy = None
        self.test_ber = None
        # for confusion matrix
        self.predictions = []
        self.truths = []

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
        self.test_accuracy_avg = None
        self.test_accuracy_std = None
        self.test_loss_avg = None
        self.test_loss_std = None
        self.predictions = []
        self.truths = []

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
        
        res.train_ber_avg = np.average(list(map(lambda x:x.train_bers, res_list)), axis=0)
        res.train_ber_std = np.std(list(map(lambda x:x.train_bers, res_list)), axis=0)
        res.valid_ber_avg = np.average(list(map(lambda x:x.valid_bers, res_list)), axis=0)
        res.valid_ber_std = np.std(list(map(lambda x:x.valid_bers, res_list)), axis=0)
        
        res.test_accuracy_avg = np.average(list(map(lambda x:x.test_accuracy, res_list)))
        res.test_accuracy_std = np.std(list(map(lambda x:x.test_accuracy, res_list)))
        res.test_loss_avg = np.average(list(map(lambda x:x.test_loss, res_list)))
        res.test_loss_std = np.std(list(map(lambda x:x.test_loss, res_list)))
        res.test_ber_avg = np.average(list(map(lambda x:x.test_ber, res_list)))
        res.test_ber_std = np.std(list(map(lambda x:x.test_ber, res_list)))
        for x in res_list:
            res.predictions.extend(x.predictions)
            res.truths.extend(x.truths)
        
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
    res.train_bers.append(balanced_accuracy_score(y_train, model.predict(X_train)))
    res.valid_bers.append(balanced_accuracy_score(y_valid, model.predict(X_valid)))
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
        res.train_bers.append(balanced_accuracy_score(y_train, model.predict(X_train)))
        res.valid_bers.append(balanced_accuracy_score(y_valid, model.predict(X_valid)))

    res.test_loss = model._loss(X_test, y_test)
    res.test_accuracy = sum(model.predict(X_test)==y_test)/len(y_test)
    res.test_ber = balanced_accuracy_score(y_test, model.predict(X_test))
    res.train_accuracies = np.array(res.train_accuracies)
    res.train_losses = np.array(res.train_losses)
    res.valid_accuracies = np.array(res.valid_accuracies)
    res.valid_losses = np.array(res.valid_losses)
    
    model.coef_ = res.best_coef
    res.truths.extend(y_test)
    res.predictions.extend(model.predict(X_test))
    return res

#############################
# All routines start below!!
#############################

def test():
    pass


############################
#
# Visualize the first four components of PCA
#
############################

def visualizePCA():
    N_COMPONENTS = 50
    N_SPLITS = 10
    
    pca = MyPCA(n_components=N_COMPONENTS)
    kfold = MyKFold(n_splits=N_SPLITS)
    i=0
    for d_train, d_valid, d_test in kfold.split_dict(images):
        i+=1
        if i!=9:continue
        X_train, y_train = d_train
        pca.fit(X_train)
        
    fig, axs = plt.subplots(2, 2)
    data_shape = 'resized' if pca.components_.shape[0]==19520 else 'aligned'
    for ax, i in zip(axs.ravel(), range(0, 4)):
        ax.set_title("Component_{}".format(str(i+1)))
        ax.set_axis_off()
        if data_shape=='resized':
            ax.imshow(np.array(pca.components_[:,i].reshape(160, 122)))
        else:
            ax.imshow(np.array(pca.components_[:,i].reshape(224, 192)))
    plt.show()

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
    LEARNING_RATE = 0.1

    kfold = MyKFold(n_splits=N_SPLITS)
    test_accuracy = 0
    test_loss = 0
    
    model = SoftmaxRegression(lr=LEARNING_RATE)
    emotions = []
    i = 0
    for d_train, d_valid, d_test in kfold.split_dict(images):
        i+=1
        if i!=9:continue
        X_train, y_train = d_train
        pca = MyPCA(n_components=N_COMPONENTS)
        pca.fit(X_train)
        result = trainModel(model, d_train, d_valid, d_test, epoch=EPOCH, n_components=N_COMPONENTS)
    
    print("test loss: {}, test accuracy: {}.".format(result.test_loss, result.test_accuracy))
    # leave out bias term
    emotion_matrix = pca.components_.dot(model.coef_[1:,]).T

    # linear scale
    emotion_matrix -= emotion_matrix.min(axis=1)
    emotion_matrix = np.multiply(emotion_matrix, 255/emotion_matrix.max(axis=1))
    emotion_matrix = emotion_matrix.astype(int)
    emotions = {}
    for i, label in enumerate(model.classes_):
        emotions[label] = np.array(emotion_matrix[i].reshape(224, 192))
    
    fig, axs = plt.subplots(2, 3)
    for ax, label in zip(axs.ravel(), emotions.keys()):
        ax.set_title(label)
        ax.set_axis_off()
        ax.imshow(emotions[label])
    plt.show()


############################
#
# Report Softmax Regression Training process
#
############################
def softmaxConfusion():
    EPOCH = 50
    LEARNING_RATE = 0.1
    res = doRegression(images, 'softmax', n_components=50, epoch=EPOCH, lr=LEARNING_RATE, useBatch=True)
    pred = res.predictions
    truth = res.truths
    dic = defaultdict(lambda:len(dic))
    for x in pred:
        dic[x]
    matrix = np.zeros((len(dic), len(dic)))
    for r, c in zip(pred, truth):
        matrix[dic[r]][dic[c]] += 1
    labels = [''] + list(map(lambda x:x[0], sorted(list(dic.items()),key=lambda x:x[1])))
    fig, ax = plt.subplots() 
    img = ax.matshow(matrix)
    ax.set_xticklabels(labels)        
    ax.set_yticklabels(labels)    
    ax.title.set_text("Softmax 10-fold Confusion Matrix")
    fig.colorbar(img, ax=ax, orientation='vertical', fraction=.1)    
    plt.show()

def softmaxSGD():
    EPOCH = 50
    LEARNING_RATE = 0.1
    gd_res = doRegression(images, 'softmax', n_components=50, epoch=EPOCH, lr=LEARNING_RATE, useBatch=True)
    sgd_res = doRegression(images, 'softmax', n_components=50, epoch=EPOCH, lr=LEARNING_RATE, useBatch=False)
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
    result = doRegression(images, 'softmax', n_components=50, epoch=EPOCH, lr=LEARNING_RATE, useBatch=True)
    
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

def doRegression(images, reg_type, n_components = 50, epoch=100, lr=0.1, useBatch=True, class_weight = None):
    # hyper parameters
    N_SPLITS = 10

    kfold = MyKFold(n_splits=N_SPLITS)
    results = []
    for d_train, d_valid, d_test in kfold.split_dict(images):
        if reg_type=='softmax':
            classifier = SoftmaxRegression(lr=lr, class_weight=class_weight)
        elif reg_type=='logistic':
            classifier = LogisticRegression(lr=lr)
        result = trainModel(classifier, d_train, d_valid, d_test, epoch=epoch, n_components=n_components, useBatch=useBatch)
        results.append(result)
    
    ret = AggregateResult.aggregate(results)
    print("TEST result: loss_avg: {:.4f} ({:.4f}), accuracy: {:.4f}({:.4f}), balanced accuracy: {:.4f}({:.4f})".format(ret.test_loss_avg, ret.test_loss_std, ret.test_accuracy_avg, ret.test_accuracy_std, ret.test_ber_avg, ret.test_ber_std))
    return ret


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
    LEARNING_RATE = 0.1
    print("----training balanced model------")
    balanced_result = doRegression(images, "softmax", n_components=50,  epoch=EPOCH, lr=LEARNING_RATE, class_weight='balanced')
    print("----training regular model------")
    reg_result = doRegression(images, 'softmax', n_components=50,  epoch=EPOCH, lr=LEARNING_RATE)
    
    xlabels = [x for x in range(0,EPOCH+1)]
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.suptitle('Softmax Regression Balanced vs Unweighted')
    axs[0].set_title('Cross-Entropy Loss')
    axs[1].set_title('Balanced Accuracy')
    axs[1].set_xlabel('Epoch')

    axs[0].errorbar(xlabels,balanced_result.valid_loss_avg,balanced_result.valid_loss_std, errorevery=10, label='balanced model loss')
    axs[0].errorbar(xlabels,reg_result.valid_loss_avg,reg_result.valid_loss_std, errorevery=10, label='unweighted model loss')
    axs[1].errorbar(xlabels,balanced_result.valid_ber_avg,balanced_result.valid_ber_std, errorevery=10, label='balanced model B-accuracy')
    axs[1].errorbar(xlabels,reg_result.valid_ber_avg,reg_result.valid_ber_std, errorevery=10, label='unweighted model B-accuracy')
    axs[0].legend()
    axs[1].legend()
    plt.show()


############################
#
# Report PCA
#
############################

def reportPCA():
    # hyper parameters
    EPOCH = 100
    N_SPLITS = 10
    LEARNING_RATE = 0.1

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

############################
#
# Report Logistic
#
############################

def reportLogisticOneRun():
    # hyper parameters
    EPOCH = 50
    N_COMPONENTS = 8
    N_SPLITS = 10
    LEARNING_RATE = 0.1
    i = 0
    kfold = MyKFold(N_SPLITS)
    for d_train, d_valid, d_test in kfold.split_dict(images):
        i += 1
        if i != 6: continue
        classifier = LogisticRegression(lr=LEARNING_RATE)
        result = trainModel(classifier, d_train, d_valid, d_test, epoch=EPOCH, n_components=N_COMPONENTS, useBatch=True)
    xlabels = [x for x in range(0,EPOCH+1)]
    plt.figure(1)
    plt.plot(xlabels,result.train_losses,color = 'b')
    plt.plot(xlabels,result.valid_losses,color = 'g')
    plt.title('resized image train loss & valid loss vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend(["train loss","valid loss"])
    plt.show()

def reportLogistic():
    # hyper parameters
    EPOCH = 50
    N_COMPONENTS = 8
    N_SPLITS = 10
    LEARNING_RATE = [0.01,0.1,5]
    balanced_results = []
    resultsets = []
    for learningrate in LEARNING_RATE:
        balanced_result = doRegression(images,'logistic', n_components=N_COMPONENTS, epoch=50, lr=learningrate, useBatch=True, class_weight = None)
        balanced_results.append(balanced_result)
        
    xlabels = [x for x in range(0,EPOCH+1)]
    plt.figure(1)
    plt.title('Train_loss vs valid_loss')
    plt.title('Cross-Entropy Loss')
    plt.xlabel('Epoch')
    plt.plot(xlabels,balanced_results[1].train_loss_avg,color = 'b')
    plt.plot(xlabels,balanced_results[1].valid_loss_avg,color = 'g')
    plt.title('aligned image train loss & valid loss vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend(["train loss","valid loss"])
    plt.figure(2)
    plt.title('Train_loss vs learning rate')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.errorbar(xlabels,balanced_results[0].train_loss_avg,balanced_results[0].train_loss_std, errorevery=10, label='train loss when lr=0.01')
    plt.errorbar(xlabels,balanced_results[1].train_loss_avg,balanced_results[1].train_loss_std, errorevery=10, label='train loss when lr=0.1')
    plt.errorbar(xlabels,balanced_results[2].train_loss_avg,balanced_results[2].train_loss_std, errorevery=10, label='train loss when lr=5')
    plt.legend(['lr = 0.001','lr= 0.1','lr = 5'])
    plt.show()


dataset, cnt = load_data(data_dir)
images = balanced_sampler(dataset, cnt, emotions=emotions)

print('--- Below is the routine output ---')
locals()[args.routine]()