import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

prep = pd.read_csv("prep.csv", header=None)
noprep = pd.read_csv("noprep.csv", header=None)

p = []
for row in prep.iterrows():
    p.append(row[1][0])

npp = []
for row in noprep.iterrows():
    npp.append(row[1][0])



plt.plot(np.arange(0, 50), p, 'r', np.arange(0, 50), npp, 'g')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend(['With preprocessing', 'W/O preprocessing'])
plt.show()

