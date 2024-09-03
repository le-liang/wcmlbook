import scipy.io as sio                     # import scipy.io for .mat file I/O
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import function_wmmse_powercontrol1 as wf   # import our function file
import function_dnn_powercontrol as df     # import our function file

K = 10                     # number of users
num_H = 25000             # number of training samples
num_test = 5000            # number of testing  samples
training_epochs = 200      # number of training epochs
trainseed = 0              # set random seed for training set
testseed = 7               # set random seed for test set

# Problem Setup
print('Gaussian IC Case: K=%d, Total Samples: %d, Total Iterations: %d\n'%(K, num_H, training_epochs))

# Generate Training Data
Xtrain, Ytrain, wtime = wf.generate_Gaussian(K, num_H, seed=trainseed)
DNN = df.PowerControl(X=Xtrain, Y=Ytrain)

# Training Deep Neural Networks
# print('train DNN ...')
# Save & Load model from this path 
model_location = "./DNNmodel/model_demok=%d.ckpt"%K
# DNN.train(model_location, training_epochs=training_epochs, batch_size=1000,  LRdecay=0)

# Generate Testing Data
X, Y, wmmsetime = wf.generate_Gaussian(K, num_test, seed=testseed)

# Testing Deep Neural Networks
dnntime = DNN.test(X, "Prediction_%d" % K, model_location, binary=0)
print('wmmse time: %0.3f s, dnn time: %0.3f s, time speed up: %0.1f X' % (wmmsetime, dnntime, wmmsetime / dnntime))

# Evaluate Performance of DNN and WMMSE
H = np.reshape(X, (K, K, X.shape[1]), order="F")
NNVbb = sio.loadmat('Prediction_%d' % K)['pred']
wf.perf_eval(H, Y, NNVbb, K)
wf.perf_eval_H(H, Y, NNVbb, K)

# Plot figures
train = sio.loadmat('MSETime_%d_%d_%d'%(K, 200, 10))['train']
time = sio.loadmat('MSETime_%d_%d_%d'%(K, 200, 10))['time']
val = sio.loadmat('MSETime_%d_%d_%d'%(K, 200, 10))['validation']
plt.figure(0)
plt.plot(time.T, val.T,label='validation')
plt.plot(time.T, train.T,label='train')
plt.legend(loc='upper right')
plt.xlabel('time (seconds)')
plt.ylabel('Mean Square Error')
# plt.savefig('MSE_train.eps', format='eps', dpi=1000)
plt.show()
