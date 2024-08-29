


def test():        
        
        training_epochs = 20
        batch_size = 256
        display_step = 5
        model_saving_step = 5
        test_step = 1000
        examples_to_show = 10

        # Network Parameters
        n_hidden_1 = 500
        n_hidden_2 = 250 # 1st layer num features
        n_hidden_3 = 120 # 2nd layer num features
        n_input = 256 # MNIST data input (img shape: 28*28)
        n_output = 16 #4
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, n_input])
        #Y = tf.placeholder("float", [None, K*mu])
        Y = tf.placeholder("float", [None, n_output])
        def encoder(x):
            weights = {                    
                'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1)),
                'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.1)),
                'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=0.1)),
                'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output],stddev=0.1)),            
            }
            biases = {            
                'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1],stddev=0.1)),
                'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2],stddev=0.1)),
                'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3],stddev=0.1)),
                'encoder_b4': tf.Variable(tf.truncated_normal([n_output],stddev=0.1)),          
            
            }
        
            # Encoder Hidden layer with sigmoid activation #1
            #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
            layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))
            return layer_4
        # Building the decoder

        #encoder_op = encoder(X)

        #for network_idx in range(0, int(K*mu/n_output)):
        #    y_pred_cur = encoder(X)
        #    if network_idx == 0:
        #        y_pred = y_pred_cur
        #    else:
        #        y_pred = tf.concat((y_pred, y_pred_cur), axis=1)            
        # Prediction
        y_pred = encoder(X)
        # Targets (Labels) are the input data.
        y_true = Y

        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        #cost = tf.reduce_mean(tf.pow(y_true - y_pred, 1))
        #cost = tf.reduce_mean(tf.abs(y_true-y_pred))
        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # Generating Detection 
        #code = BinaryLinearBlockCode(parityCheckMatrix='./test/data/BCH_63_36_5_strip.alist')
        #code = PolarCode(6, SNR=4, mu = 16, rate = 0.5)
        #decoders = [IterativeDecoder(code, minSum=True, iterations=50, reencodeOrder=-1, reencodeRange=0.1)]        

        # Start Training
        config_GPU = tf.ConfigProto()
        config_GPU.gpu_options.allow_growth = True
        # The H information set
        test_idx_low = 1
        test_idx_high = 80      
        '''
        H_folder = '../H_dataset/'
        test_idx_low = 301
        test_idx_high = 400 
        '''
        channel_response_set_test = []
        for test_idx in range(test_idx_low,test_idx_high):
            H_file = H_folder + str(test_idx) + '.txt'
            with open(H_file) as f:
                for line in f:
                    numbers_str = line.split()
                    numbers_float = [float(x) for x in numbers_str]
                    h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                    channel_response_set_test.append(h_response)




        print ('length of testing channel response', len(channel_response_set_test))



        saver = tf.train.Saver()
        
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)            
            saving_name = config.model_name
            saver.restore(sess, saving_name)            
            input_samples_test = []
            input_labels_test = []
            test_number = 100000        
            for i in range(0, test_number):
                bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, )) 
                #signal_train, signal_output, para = ofdm_simulate(bits) 
                #codeword = code.encode(bits)    
                #signal_train, signal_output, para = ofdm_simulate(codeword) 
                channel_response= channel_response_set_test[np.random.randint(0,len(channel_response_set_test))]
                #signal_output, para = ofdm_simulate_single(bits,channel_response)
                signal_output, para = ofdm_simulate(bits,channel_response,SNRdb)
                #input_labels_test.append(codeword)
                input_labels_test.append(bits[config.pred_range])
                #input_samples_test.append(np.concatenate((signal_train,signal_output)))
                input_samples_test.append(signal_output)
                        
            batch_x = np.asarray(input_samples_test)
            batch_y = np.asarray(input_labels_test)
            encode_decode = sess.run(y_pred, feed_dict = {X:batch_x})
            mean_error = tf.reduce_mean(abs(y_pred - batch_y))                
            BER = 1-tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.equal(tf.sign(y_pred-0.5), tf.cast(tf.sign(batch_y-0.5),tf.float32))),1))
                        
            print("OFDM Detection QAM output number is", n_output, "SNR = ", SNRdb, "Num Pilot", P,"prediction and the mean error on test set are:", mean_error.eval({X:batch_x}), BER.eval({X:batch_x}))


