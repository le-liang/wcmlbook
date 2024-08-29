import scipy.io as sio                     # import scipy.io for .mat file I/O
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import function_wmmse_powercontrol as wf   # import our function file
import ePCNet as df     # import our function file

K = 10                     # number of users
num_H = 10000              # number of training samples
num_test = 500            # number of testing  samples
training_epochs = 1000     # number of training epochs
trainseed = 0              # set random seed for training set
testseed = 0               # set random seed for test set

# Problem Setup
print('Gaussian IC Case: K=%d, Total Samples: %d, Total Iterations: %d\n'%(K, num_H, training_epochs))

# Generate Training Data
Xtrain, Ytrain, wtime = wf.generate_Gaussian(K, num_H, seed=trainseed)
DNN = df.PowerControl(X=Xtrain, Y=Ytrain)

# Training Deep Neural Networks
#print('train DNN ...')
# Save & Load model from this path 
model_location = "./DNNmodel/model_demok_qos=%d.ckpt"%K
DNN.train(model_location, training_epochs=training_epochs, batch_size=1000,  LRdecay=0)

# Generate Testing Data
X, Y, wmmsetime = wf.generate_Gaussian(K, num_test, seed=testseed)

# Testing Deep Neural Networks

# print('wmmse time: %0.3f s, dnn time: %0.3f s, time speed up: %0.1f X' % (wmmsetime, dnntime, wmmsetime / dnntime))

# Evaluate Performance of DNN and WMMSE
H = np.reshape(X, (K, K, X.shape[1]), order="F")
nnrate = DNN.test(H, X, "Prediction_qos%d" % K, model_location, binary=0)
# NNVbb = sio.loadmat('Prediction_qos%d' % K)['pred']
wf.perf_eval(H, Y, nnrate, K)

# Plot figures
# train = sio.loadmat('MSETime_%d_%d_%d'%(K, 200, 10))['train']
# time = sio.loadmat('MSETime_%d_%d_%d'%(K, 200, 10))['time']
# val = sio.loadmat('MSETime_%d_%d_%d'%(K, 200, 10))['validation']
# plt.figure(0)
# plt.plot(time.T, val.T,label='validation')
# plt.plot(time.T, train.T,label='train')
# plt.legend(loc='upper right')
# plt.xlabel('time (seconds)')
# plt.ylabel('Mean Square Error')
# plt.savefig('MSE_train.eps', format='eps', dpi=1000)
# plt.show()
