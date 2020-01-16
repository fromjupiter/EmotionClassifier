import numpy as np
from dataloader import load_data, balanced_sampler
from collections import defaultdict

class MyKFold(object):
    def __init__(self, n_splits=10):
        self.n_splits = n_splits

    # 
    # split a dictionary-like dataset
    # for label in labels:
    #   dataset[label1] = list of 2-dimension images
    #
    # return list of ((X_train,y_train), (X_valid, y_valid), (X_test, y_test))
    def split_dict(self, dic):
        i = 0
        data = {}
        data_dim = None
        for label, images in dic.items():
            if len(images)>0:
                data_dim = images[0].shape[0]*images[1].shape[1]
            data[label] = np.array_split(list(map(lambda x:x.flatten(), images)), self.n_splits)
        meta = {}
        for i in range(self.n_splits):
            X_train = []
            y_train = []
            X_valid = []
            y_valid = []
            X_test = []
            y_test = []
            for label, splits in data.items():
                valid_slice = i
                test_slice = (i+1)%len(splits)
                train_slice1 = slice(0 if i<len(splits)-1 else 1, i)
                train_slice2 = slice(i+2, None)
                X_valid.extend(splits[valid_slice])
                y_valid.extend([label]*len(splits[valid_slice]))
                X_test.extend(splits[test_slice])
                y_test.extend([label]*len(splits[test_slice]))
                for x in splits[train_slice1]:
                    X_train.extend(x)
                    y_train.extend([label]*len(x))
                for x in splits[train_slice2]:
                    X_train.extend(x)
                    y_train.extend([label]*len(x))
            X_train = np.matrix(X_train)
            y_train = np.array(y_train)
            X_valid = np.matrix(X_valid)
            y_valid = np.array(y_valid)
            X_test = np.matrix(X_test)
            y_test = np.array(y_test)
            yield ((X_train, y_train), (X_valid, y_valid), (X_test, y_test))
            
            

if __name__=='__main__':
    IMG_NUM = 40
    N_SPLITS = 10
    CLASSES = 3
    test = {str(i):[np.ones((5,5))*i]*IMG_NUM for i in range(1, CLASSES+1)}
    
    kf = MyKFold(N_SPLITS)
    count = 0
    for train, valid, test in kf.split_dict(test):
        X_train, y_train = train
        X_valid, y_valid = valid
        X_test, y_test = test
        assert len(X_train)==len(y_train)
        assert len(X_valid)==len(y_valid)
        assert len(X_test)==len(y_test)
        assert len(X_train)==int((IMG_NUM*CLASSES)/N_SPLITS*(N_SPLITS-2))
        assert len(X_valid)==int((IMG_NUM*CLASSES)/N_SPLITS)
        assert len(X_test)==int((IMG_NUM*CLASSES)/N_SPLITS)
        count+=1
    
    assert count==N_SPLITS