from PCA import MyPCA
from SoftmaxRegression import SoftmaxRegression
from kfold import MyKFold
from dataloader import load_data, balanced_sampler

data_dir = "../aligned/"
dataset, cnt = load_data(data_dir)
images = balanced_sampler(dataset, cnt, emotions=['happiness','anger'])

# hyper parameters
N_COMPONENTS = 5
N_SPLITS = 10
LEARNING_RATE = 0.01

kfold = MyKFold(n_splits=10)

i=-1
for d_train, d_valid, d_test in kfold.split_dict(images):
    i+=1
    # test only 9th split
    if i!=9: 
        continue
    X_train, y_train = d_train
    X_valid, y_valid = d_valid
    X_test, y_test = d_test
    pca = MyPCA(n_components=N_COMPONENTS)
    pca.fit(X_train)
    Xpca_train = pca.transform(X_train)
    classifier = SoftmaxRegression(lr=LEARNING_RATE)
    classifier.fit(Xpca_train, y_train)
    pred = classifier.predict(pca.transform(X_valid))
    # print(pred)
    # print(y_valid)
    # print(sum(pred==y_valid)/len(y_valid))

    
    pred = classifier.predict(Xpca_train)
    # print(classifier.predict_proba(Xpca_train))
    # print(classifier.classes_)
    # print(pred)
    # print(y_train)
    print(classifier.predict_proba(Xpca_train))
    print(classifier._encode_y(y_train))
    print(sum(pred==y_train)/len(y_train))
    