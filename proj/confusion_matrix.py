import numpy as np
import matplotlib.pyplot as plt

def confusion(predict,truth):
    predict = predict.argmax(axis = 1)
    truth = truth.argmax(axis = 1)
    nums = len(truth)
    mat = np.zeros([nums,nums])
    for i in range(nums):
        mat[truth[i],predict[i]] += 1
    correct = np.sum(mat.diagonal())
    mat /= np.reshape(np.sum(mat, axis=1),(nums,1))
    acc = correct/nums*100
    print('x=truth,y = prediction')
    plt.imshow(mat)
    print(acc)
    plt.show()

def plotloss(loss,std,ylabel = 'train_loss'):
    plt.plot(loss)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.title('epoch vs'+ ylabel)





if __name__ == '__main__':
    predict = np.array([[1,0,0],[0,1,0],[0,0,1]])
    truth = np.array([[1,0,0],[0,1,0],[0,0,1]])
    mat,acc = confusion(predict,truth)
    print('x=truth,y = prediction')
    plt.imshow(mat)
    print(acc)
    plt.show()
