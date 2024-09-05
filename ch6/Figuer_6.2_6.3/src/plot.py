import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


import csv


data = pd.read_csv(r'./save/1.csv')
y = []
for i, row in enumerate(data):
    if i < 5:  
        x = row[1:]
        y.append(x)


# Plot Loss curve
plt.figure()
plt.title('Training Loss vs Communication rounds')
plt.plot(y[0],y[1],color='r', linestyle='--')
plt.plot(y[0],y[2],color='r', linestyle='-')
plt.plot(y[0],y[3],color='b', linestyle='--')
plt.plot(y[0],y[4],color='b', linestyle='-')
plt.legend('fedavg-iid','fedavg-noniid','fedsgd-iid','fedsgd-noniid')
plt.ylabel('Training loss')
plt.xlabel('Communication Rounds')
plt.savefig('1.eps')
plt.show()
