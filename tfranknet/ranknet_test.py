import os
import numpy as np
from ranknet import RankNet
from sklearn.cross_validation import cross_val_predict

os.system("rm -rf testlog")
data1 = np.random.rand(1000, 30)
data2 = np.random.rand(1000, 30)
label = [True]*1000

rn = RankNet(hidden_units=[20, 10],
             learning_rate=0.01)
data = rn.pack_data(data1, data2)
rn.pretrain(data1)

rn.fit(data, logdir="logfine")

score = rn.get_scores(data1)


if False:
    cvpred = cross_val_predict(rn, data, label, cv=2)
    input1 = np.random.rand(10, 30)
    input2 = np.random.rand(10, 30)
    input_ = rn.pack_data(input1, input2)
    prob = rn.predict_prob(input_)
    pred = rn.predict(input_)
    score = rn.get_scores(input1)
    score = rn.get_scores(input2)
