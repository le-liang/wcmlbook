from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.linalg import fractional_matrix_power, toeplitz, sqrtm
""" This file is a trial for training codes under Rayleigh fading channel """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print(os.environ["CUDA_VISIBLE_DEVICES"])
''' ========= functions ========= '''

def encoder_multiuser(x):
    with tf.variable_scope("encoding", reuse=tf.AUTO_REUSE):
        def encoding_sub(x):
            x = tf.reshape(x, [-1, block_length, 1])
            conv1 = tf.layers.conv1d(inputs=x, filters=256, kernel_size=5, padding='same')
            conv1 = tf.nn.relu(conv1)
            conv2 = tf.layers.conv1d(inputs=conv1, filters=128, kernel_size=3, padding='same')
            conv2 = tf.nn.relu(conv2)
            conv3 = tf.layers.conv1d(inputs=conv2, filters=64, kernel_size=3, padding='same')
            conv3 = tf.nn.relu(conv3)
            conv4 = tf.layers.conv1d(inputs=conv3, filters=2, kernel_size=3, padding='same')
            layer_4_normalized = tf.scalar_mul(tf.sqrt(tf.cast(0.5*block_length, tf.float32)),
                                                      tf.nn.l2_normalize(conv4, dim=1))  # normalize the encoding.
            return layer_4_normalized

        output = tf.concat([encoding_sub(x[:,:,i]) for i in range(Tx_num)], -1)   # concat along the third dimension
        #print ("output of encoding", output)
        return output

def encoder(x):
    with tf.variable_scope("encoding", reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv1d(inputs=x, filters=256, kernel_size=5, padding='same')
        conv1 = tf.nn.relu(conv1)
        conv2 = tf.layers.conv1d(inputs=conv1, filters=128, kernel_size=3, padding='same')
        conv2 = tf.nn.relu(conv2)
        conv3 = tf.layers.conv1d(inputs=conv2, filters=64, kernel_size=3, padding='same')
        conv3 = tf.nn.relu(conv3)
        conv4 = tf.layers.conv1d(inputs=conv3, filters=2*Tx_num, kernel_size=3, padding='same')
        layer_4_normalized = tf.scalar_mul(tf.sqrt(tf.cast(block_length/2, tf.float32)),
                                                   tf.nn.l2_normalize(conv4, dim=1))  # normalize the encoding.
        return layer_4_normalized


def decoder(x, side_info):
    with tf.variable_scope("decoding", reuse=tf.AUTO_REUSE):
        # a bilinear layer of x and side_info:
        def bilinear_mul(x, info):
            x_reshape = tf.reshape(x, [-1, block_length*2*Rx_num, 1])
            info_reshape = tf.reshape(info, [-1, 1, len_info])
            mat_reshape = tf.matmul(x_reshape, info_reshape)
            return tf.reshape(mat_reshape, [-1, block_length, len_info*2*Rx_num])
        x_combine = bilinear_mul(x, side_info)
        conv1 = tf.layers.conv1d(inputs=x_combine, filters=256, kernel_size=5, padding='same')
        conv1 = tf.nn.relu(conv1)
        conv2 = tf.layers.conv1d(inputs=conv1, filters=128, kernel_size=5, padding='same')
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.conv1d(inputs=conv2, filters=128, kernel_size=5, padding='same')
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.conv1d(inputs=conv2, filters=128, kernel_size=5, padding='same')
        conv2 = tf.nn.relu(conv2)
        conv3 = tf.layers.conv1d(inputs=conv2, filters=64, kernel_size=5, padding='same')
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.layers.conv1d(inputs=conv3, filters=64, kernel_size=5, padding='same')
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.layers.conv1d(inputs=conv3, filters=64, kernel_size=3, padding='same')
        conv3 = tf.nn.relu(conv3)
        conv4 = tf.layers.conv1d(inputs=conv3, filters=32, kernel_size=3, padding='same')
        conv4 = tf.nn.relu(conv4)
        Decoding_logit = tf.layers.conv1d(inputs=conv4, filters=Tx_num, kernel_size=3, padding='same')
        Decoding_prob = tf.nn.sigmoid(Decoding_logit)
        return Decoding_logit[:, 0:block_length, :], Decoding_prob[:, 0:block_length, :]



def ChannelEstimation(x):
    with tf.variable_scope("Channelestimation", reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv1d(inputs=x, filters=256, kernel_size=5, padding='same')
        conv1 = tf.nn.relu(conv1)
        conv2 = tf.layers.conv1d(inputs=conv1, filters=128, kernel_size=3, padding='same')
        conv2 = tf.nn.relu(conv2)
        conv3 = tf.layers.conv1d(inputs=conv2, filters=64, kernel_size=3, padding='same')
        conv3 = tf.nn.relu(conv3)
        conv4 = tf.layers.conv1d(inputs=conv3, filters=32, kernel_size=3, padding='same')
        conv4 = tf.nn.relu(conv4)
        conv4 = tf.layers.conv1d(inputs=conv4, filters=3, kernel_size=3, padding='same')
        conv4 = tf.nn.relu(conv4)
        conv4_flat = tf.reshape(conv4, [-1, 3*(block_length+len_h)])
        FC = tf.nn.relu(tf.layers.dense(conv4_flat, 100, activation = None))
        output = tf.layers.dense(FC, len_info, activation=None)

        return output

def MIMO_Rayleigh_layer(x, h, std):
    h_r = h[:,:,:,0]  # get the real part
    h_i = h[:,:,:,1]  # get the imagnary part
    x_r = x[:,:,::2]
    x_i = x[:,:,1::2]
    output_r = tf.matmul(x_r, h_r) - tf.matmul(x_i, h_i)
    output_i = tf.matmul(x_r, h_i) + tf.matmul(x_i, h_r)
    output = tf.concat([output_r, output_i], 2)
    noise = tf.random_normal(shape=tf.shape(output), mean=0.0, stddev=std, dtype=tf.float32)
    return output+noise


def Rayleigh_layer(x, h, std):
    output_r = tf.reshape(x[:, :, 0] * tf.reshape(h[:, 0], [-1, 1]) - x[:, :, 1] * tf.reshape(h[:, 1], [-1, 1]),
                          [-1, block_length, 1])
    output_i = tf.reshape(x[:, :, 0] * tf.reshape(h[:, 1], [-1, 1]) + x[:, :, 1] * tf.reshape(h[:, 0], [-1, 1]),
                          [-1, block_length, 1])
    output = tf.concat([output_r, output_i], -1)
    noise = tf.random_normal(shape=tf.shape(output), mean=0.0, stddev=std, dtype=tf.float32)
    return output + noise

def Multipath_layer(x, h, std):
    # This layer simulate the multipath layer
    x_pad = tf.pad(x, tf.constant([[0, 0], [0, len_h], [0, 0]]))

    h_r = tf.reshape(h[:, :, 0], [-1, len_h, 1])
    h_i = tf.reshape(h[:, :, 1], [-1, len_h, 1])
    x_r = tf.reshape(x_pad[:, :, 0], [-1, block_length + len_h, 1])
    x_i = tf.reshape(x_pad[:, :, 1], [-1, block_length + len_h, 1])

    def convolution(x, h):
        y = x * tf.reshape(h[:, 0, 0], [-1, 1, 1])
        for i in range(1, len_h):
            cur = x * tf.reshape(h[:, i, 0], [-1, 1, 1])
            cur = tf.concat([cur[:, -i:, :], cur[:, :-i, :]], 1)
            y += cur
        return y

    o_r = convolution(x_r, h_r) - convolution(x_i, h_i)
    o_i = convolution(x_r, h_i) + convolution(x_i, h_r)
    output = tf.concat([o_r, o_i], -1)
    noise = tf.random_normal(shape=tf.shape(output), mean=0.0, stddev=std, dtype=tf.float32)
    output += noise
    return output


def generate_batch_data(batch_size):
    global start_idx, data
    if start_idx + batch_size >= N_train:
        start_idx = 0
        data = np.random.binomial(1, 0.5, [N_train, block_length, Tx_num])
    batch_x = data[start_idx:start_idx + batch_size]
    start_idx += batch_size
    # print("start_idx", start_idx)
    return batch_x

def sample_h_mimo(sample_size):
    h = 1 / np.sqrt(2) * np.random.normal(size=sample_size)

    if rho > 0: # correlated channel
        R_tx = toeplitz([rho**(i*i) for i in range(Tx_num)])
        R_rx = toeplitz([rho**(i*i) for i in range(Rx_num)])
        h_complex = h[:,:,:,0] + 1j*h[:,:,:,1]
        for i in range(len(h)):
            h_complex_correlated = sqrtm(R_rx).dot(h_complex[i,:,:]).dot(sqrtm(R_tx))
            h[i,:,:,0], h[i,:,:,1] = np.real(h_complex_correlated), np.imag(h_complex_correlated)

    return h


''' ========= Main function begins here: ======== '''

batch_size = 512
Tx_num = 4    # number of tx antennas or the number of users
Rx_num = 4    # number of rx antennas in the BS
M = 64        # Input length
N_train = 160000
data = np.random.binomial(1, 0.5, [N_train, M, Tx_num])
N_val = 1500
val_data = np.random.binomial(1, 0.5, [N_val, M, Tx_num])
N_test = 5000
test_data = np.random.binomial(1, 0.5, [N_test, M, Tx_num])
X = tf.placeholder("float", [None, M, Tx_num])
Noise_std = tf.placeholder("float", [])
block_length = M
EbNo_train = 15
EbNo_train = 10 ** (EbNo_train / 10)
EbNo_test = 15
EbNo_test = 10 ** (EbNo_test / 10)
len_h = 0
len_info = 60
batch_size_h = 1
rho = 0.7
#PDP = generate_PDP(len_h)
h = tf.placeholder(tf.float32, shape=[None, Tx_num, Rx_num, 2])
X = tf.placeholder("float", [None, M, Tx_num])
Noise_std = tf.placeholder("float", [])
encoding = encoder(X)
encoding_noised = MIMO_Rayleigh_layer(encoding, h, Noise_std)
SideInfo = ChannelEstimation(encoding_noised)
Y_logit, Y = decoder(encoding_noised, SideInfo)

learning_rate = 0.0001
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_logit, labels=X))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
tvs = tf.trainable_variables()
accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
gvs = optimizer.compute_gradients(loss_op, tvs)
accum_ops = [tf.assign_add(accum_vars[i], gv[0]) for i, gv in enumerate(gvs)]
train_op = optimizer.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])
accuracy = 1 - tf.reduce_mean(tf.cast(tf.abs(Y - X) < 0.5, tf.float32))
R = 0.5
number_steps = int(5e6) + 1
testing_step = 1000
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()
saver = tf.train.Saver()
model_saving_step = int(1e5)
load_pretrain_model = True
with tf.Session(config=config) as sess:
    sess.run(init)
    start_idx = 0
    print("start training")
    for step in range(number_steps):
        if step % 100 == 0:
            print("Training step:", step)
        batch_x = generate_batch_data(batch_size)
        sess.run(zero_ops)
        for i in range(batch_size_h):
            sess.run(accum_ops, feed_dict={X: batch_x, h: sample_h_mimo([batch_size, Tx_num, Rx_num, 2]), Noise_std: (np.sqrt(1/(2*R*EbNo_train)))})
        sess.run(train_op)
        if step%testing_step == 0 and step > 0:
            EbNodB_range = np.arange(0, 31, 5)
            ber = np.ones(len(EbNodB_range))
            for n in range(0, len(EbNodB_range)):
                EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
                ber_list = list()
                idx = 0
                while idx + batch_size < N_test:
                    ber_list.append(1-sess.run(accuracy, feed_dict={X:test_data[idx:idx+batch_size], h: sample_h_mimo([batch_size, Tx_num, Rx_num, 2]),  Noise_std: (np.sqrt(1/(2*R*EbNo)))}))
                    idx += batch_size
                ber[n]=1-np.mean(np.asarray(ber_list))
                print('SNR:', EbNodB_range[n], 'BER:', ber[n])
            plt.plot(EbNodB_range, ber, 'bo', label='Convolution Autoencoder')
            plt.yscale('log')
            plt.xlabel('EbN0')
            plt.ylabel('Bit Error Rate')
            plt.grid()
            plt.grid(b=True, which='major', linestyle='-')
            plt.grid(b=True, which='minor', linestyle='--')
            plt.legend(loc='upper right', ncol=1)
            plt.savefig('./images/deepcode_1d_conv_64_128_' + '{}.png'.format(str(step).zfill(3)), bbox_inches='tight')
            plt.clf()
        if step % model_saving_step == 0 and step > 0:
            save_path = saver.save(sess, './MIMO_Models_rho_7/MIMO_model_step_'+str(step)+'.ckpt')
