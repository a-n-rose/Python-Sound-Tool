from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LSTM, MaxPooling2D, \
    Dropout, TimeDistributed, ConvLSTM2D


def assign_model_settings(num_labels,sparse_targets):
    if num_labels <= 2:
        loss_type = 'binary_crossentropy'
        # binary = "sigmoid"; multiple classification = "softmax"
        activation_output = 'sigmoid' 
    else:
        loss_type = 'categorical_crossentropy'
        # binary = "sigmoid"; multiple classification = "softmax"
        activation_output = 'softmax' 
    if sparse_targets:
        # if data have mutiple labels which are only integer encoded, *not* one hot encoded.
        loss_type = 'sparse_categorical_crossentropy' 
    return loss_type, activation_output


def buildmodel(model_type,num_labels,frame_width,timesteps,num_features,
               color_scale,lstm_cells,feature_map_filters,kernel_size,
               pool_size,dense_hidden_units,activation_output):
    if 'lstm' == model_type.lower():
        model = Sequential()
        model.add(LSTM(lstm_cells,return_sequences=True,
                       input_shape = (frame_width, num_features))) 
        model.add(LSTM(lstm_cells,return_sequences=True))   
        
    elif 'cnn' == model_type.lower():
        model = Sequential()
        # 4x8 time-frequency filter (goes along both time and frequency axes)
        model.add(Conv2D(feature_map_filters, 
                         kernel_size=kernel_size, 
                         activation='relu',
                         input_shape = (frame_width * timesteps, num_features,color_scale)))
        #non-overlapping pool_size 3x3
        model.add(MaxPooling2D(pool_size = pool_size))
        model.add(Dropout(0.25))
        model.add(Dense(dense_hidden_units))
        
    elif 'cnnlstm' == model_type.lower():
        cnn = Sequential()
        cnn.add(Conv2D(feature_map_filters, kernel_size=kernel_size, activation='relu'))
        # non-overlapping pool_size 3x3
        cnn.add(MaxPooling2D(pool_size=pool_size))
        cnn.add(Dropout(0.25))
        cnn.add(Flatten())

        # prepare stacked LSTM
        model = Sequential()
        model.add(TimeDistributed(cnn,input_shape = (timesteps, frame_width,
                                                     num_features, color_scale)))
        model.add(LSTM(lstm_cells,return_sequences=True))
        model.add(LSTM(lstm_cells,return_sequences=True))

    model.add(Flatten())
    model.add(Dense(num_labels,activation = activation_output)) 

    return model
