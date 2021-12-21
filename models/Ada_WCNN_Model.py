#%%
# Read imports and utility functions
exec(open('import.py').read())

## The wda value determines which model you want to run.
wda = "Ada_WCNN"

## Scale the data
scaler = StandardScaler()
x_train2 = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
x_test2 = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
y_train2 = data.y_train
y_test2 = data.y_test
num_classes = data.nclasses

##
def pooling_type(wda):
    x = AveragePooling1D(strides=2)
    return(x)

## CNN block creation based on model type
def wd_layers(nr_blocks,first_layer,seq_len,sens,stride,kernel,wda,first_kernel_size,first_stride):
    model_list = []
    input_shape2 = []
    for i in range(nr_blocks):
        #print(i)
        
        input_shape = Input((seq_len,sens))
        xx = Conv1D(filters=first_layer,kernel_size=first_kernel_size,strides=first_stride,padding='same')(input_shape)
        xx = BatchNormalization()(xx)
        xx = Activation('relu')(xx)
        xx = pooling_type(wda)(xx)
        xx = Dropout(float(sys.argv[6]))(xx)
        print(type(wda))
        print(wda)
        model_list.append(xx)
        input_shape2.append(input_shape)
    print(len(model_list))
    print(len(input_shape2))
    return(model_list,input_shape2)
    

# Full Ada_WCNN model function
def full_model_WDMTCNN(x_train2,x_test2,y_train2,y_test2):
    pooling = pooling_type(wda)

    # Number of sensors and number of CNN blocks
    sens = x_test2.shape[2]
    nr_blocks = sens

    #Create dummies for the labels
    y_train2 = pd.DataFrame(y_train2, columns=['label'])
    dummies = pd.get_dummies(y_train2['label']) # Classification
    products = dummies.columns
    y = dummies.values
    
    # Split multivariate signals into separate time series
    datax_test0 = np.dsplit(x_test2,nr_blocks)

     # Sequence length
    seq_len = datax_test0[0].shape[1]

    # Calculate the properties of the first convolutional layer, e.g., kernel size, stride and number of filters.
    first_kernel_size = 64
    first_stride = int(first_kernel_size/4)
    penalty = (seq_len / first_stride)
    first_layer = int((seq_len * sens) / penalty)

    # Initialize k-folds
    kf = StratifiedKFold(5, shuffle=False) # Use for StratifiedKFold classification
    fold = 0

    # Build empty lists for results
    oos_y = []
    oos_pred = []
    oos_test_pred = []
    oos_test_y = []
    oos_test_prob = []
    
    # Earlystopping callback
    earlystop = EarlyStopping(monitor= 'val_loss', min_delta=0 , patience=int(sys.argv[7]), verbose=0, mode='auto')
    
    # Initialize loop for every kth fold
    for train, test in kf.split(x_train2, y_train2['label']): # Must specify y StratifiedKFold for 
        model_checkpoint = ModelCheckpoint('WD_CWRU_model.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
        fold+=1
        print(f"Fold #{fold}")
        x_train = x_train2[train]
        y_train = y[train]
        x_test = x_train2[test]
        y_test = y[test]
        
        x_train0 = np.dsplit(x_train, nr_blocks)
        x_test0 = np.dsplit(x_test,nr_blocks)

        xx,input_shape = wd_layers(nr_blocks,first_layer,seq_len,sens,1,3,wda,first_kernel_size,first_stride)
        if len(xx) > 1:
            xx = concatenate([k for k in xx])
        else:
            xx = xx[0]
        
        xx = Conv1D(filters=32,kernel_size=3,strides=1,padding='same')(xx)
        xx = BatchNormalization()(xx)
        xx = Activation('relu')(xx)
        xx = pooling_type(wda)(xx)
        xx = Dropout(float(sys.argv[6]))(xx)
        
        ls = ['same','same','valid']
        for j in ls:
            xx = Conv1D(filters=64,kernel_size=3,strides=1,padding=j)(xx)
            xx = BatchNormalization()(xx)
            xx = Activation('relu')(xx)
            xx = pooling_type(wda)(xx)
            xx = Dropout(float(sys.argv[6]))(xx)

        xx = Flatten()(xx)
        xx = Dense(100, activation = 'sigmoid')(xx)
        xx = Dropout(0.5)(xx)
        output = Dense(num_classes, activation = "sigmoid")(xx)

        # Create combined model
        wdcnn_model = Model(inputs=input_shape,outputs=output)
        print(wdcnn_model.summary())
        nr_params = wdcnn_model.count_params()

        # initialize optimizer and random generator within one fold
        keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=156324)
        keras.optimizers.SGD(lr=0.01)
        wdcnn_model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])

        ## Fit the model
        wdcnn_model.fit([i for i in x_train0], y_train,validation_data = ([j for j in x_test0],y_test), epochs = 300, batch_size = 32, verbose=1, 
                      callbacks =[earlystop,model_checkpoint], shuffle = True)
        
        ## Load best model
        wdcnn_model = load_model('./WD_CWRU_model.hdf5',custom_objects={'attention': attention})

        # Predictions on the validation set
        predictions = wdcnn_model.predict([j for j in x_test0])

        # Append actual labels of the validation set to empty list
        oos_y.append(y_test)
        # Raw probabilities to chosen class (highest probability)
        predictions = np.argmax(predictions,axis=1)
        # Append predictions of the validation set to empty list
        oos_pred.append(predictions)  
        
        # Measure this fold's accuracy on validation set compared to actual labels
        y_compare = np.argmax(y_test,axis=1) 
        score = metrics.accuracy_score(y_compare, predictions)
        print(f"Validation fold score(accuracy): {score}")
    	
    	# Predictions on the test set
        test_predictions_loop = wdcnn_model.predict([k for k in datax_test0])

        # Append actual labels of the test set to empty list
        oos_test_y.append(y_test2)
        # Append raw probabilities of the test set to empty list
        oos_test_prob.append(test_predictions_loop)
        # Raw probabilities to chosen class (highest probability)
        test_predictions_loop = np.argmax(test_predictions_loop, axis=1)
        # Append predictions of the test set to empty list
        oos_test_pred.append(test_predictions_loop)
        
        # Measure this fold's accuracy on test set compared to actual labels
        test_score = metrics.accuracy_score(y_test2, test_predictions_loop)
        print(f"Test fold score (accuracy): {test_score}")
        
    # Build the prediction list across all folds
    oos_y = np.concatenate(oos_y)
    oos_pred = np.concatenate(oos_pred)
    oos_y_compare = np.argmax(oos_y,axis=1) 

    # Measure aggregated accuracy across all folds on the validation set
    aggregated_score = metrics.accuracy_score(oos_y_compare, oos_pred)
    print(f"Aggregated validation score (accuracy): {aggregated_score}")    
    
    # Build the prediction list across all folds
    oos_test_y = np.concatenate(oos_test_y)
    oos_test_pred = np.concatenate(oos_test_pred)
    oos_test_prob = np.concatenate(oos_test_prob)
    
    # Measure aggregated accuracy across all folds on the test set
    aggregated_test_score = metrics.accuracy_score(oos_test_y, oos_test_pred)
    print(f"Aggregated test score (accuracy): {aggregated_test_score}")

    end = time.time()
    runtime = exec_time(start,end)
    
    return(oos_test_prob, oos_test_y, aggregated_score, aggregated_test_score, runtime, earlystop.patience, 
           oos_test_y,nr_params,first_layer,first_stride,penalty)

# Initialize the full_model_WDMTCNN function
oos_test_y = []
oos_test_prob = []
aggregated_score = 0
aggregated_test_score = 0
earlystop = 0
oos_test_y = []
   

oos_test_prob, oos_test_y, aggregated_score, aggregated_test_score, earlystop, oos_test_y,nr_params,first_layer,first_stride,penalty = full_model_WDMTCNN(x_train2,x_test2,y_train2,y_test2)

# %%
ls = []
ls.append([aggregated_score, aggregated_test_score, nr_params, earlystop,runtime,sys.argv[8],first_stride,penalty, 
           first_layer,num_classes,sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7]])




#%%
