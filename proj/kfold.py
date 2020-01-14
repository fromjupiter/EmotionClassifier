from sklearn.model_selection import KFold
import numpy as np
from dataloader import load_data, balanced_sampler


data_dir = "./aligned/"
dataset, cnt = load_data(data_dir)
# test with happiness and anger
images = balanced_sampler(dataset, cnt, emotions=['happiness', 'anger'])

