from matplotlib import pyplot as plt
import pickle
import numpy as np

path="data/video5.mp4"

with open("all_delta{}".format(path.split("/")[1].split('.')[0]),'rb') as f:
    all_delta=pickle.load(f)

deltas=np.array([x for x in zip(*[x for x in enumerate(all_delta)])][1])[1:5100]
i=np.array([x for x in zip(*[x for x in enumerate(all_delta)])][0])[1:5100]
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.scatter(i,deltas,s=0.5)
plt.show()