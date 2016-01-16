import os
import numpy as np
from ranknet import RankNet

os.system("rm -rf testlog")
data1 = np.random.rand(10000, 30)
data2 = np.random.rand(10000, 30)

rn = RankNet(hidden_units=[20, 10], logdir="testlog",
             learning_rate=0.01)
rn.fit(data1, data2)

prob = rn.predict_prob(data1, data2)
pred = rn.predict(data1, data2)
