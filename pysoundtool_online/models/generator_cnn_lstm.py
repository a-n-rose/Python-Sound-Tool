import numpy as np


#feed data to models
class Generator:
    def __init__(self,model_type,data,timesteps,frame_width):
        '''
        This generator pulls data out in sections (i.e. batch sizes).
        
        It then prepares that batched data to be the right shape for the 
        following models:
        * CNN or convolutional neural network
        * LSTM or long short-term memory neural network
        * CNN+LSTM, the LSTM stacked ontop of the CNN
        
        These operations performed on small batches allows you to train an 
        algorithm with quite a lot of data without much memory cost. 
        ~ trying to adjust a full dataset will certainly stall most computers ~
        
        This generator pulls data out from just one recording/word at a time.
        We define this when we extracted the features: each recording/word
        gets 5 sets (timesteps) of 11 frames (framewidth).
        
        The batch size is calculated by multiplying the frame width (e.g. 11) with 
        the timestep (e.g. 5). This is the total number of samples alloted 
        each recording/word.
        
        The label of the recording/word is in the last column of the data.
        
        Once the features and the labels are separated, each are shaped 
        according to the model they will be fed to.
        
        CNNs are great with images. Images have heights, widghts, color scheme.
        This is the shape CNNs need: (h,w,c)
        h = number of pixels up
        w = number of pixels wide
        c = whether it is grayscale (i.e. 1), rgb (i.e. 3), or rgba (i.e.4)
        * because our data is not really in color, 1 is just fine.
        
        LSTMs deal with series of data. They want to have things in timesteps.
        (timestep,....) and whatever other data you wanna put in there. We 
        have the number of frames with number of features:
        (timestep, num_frames, num_features)
        Notice here no color scheme number is needed.
        
        CNN+LSTM is a mixture of both. So a mixture of the dimensions is necessary.
        When the data is fed to the network, first it is fed to a "Time Distribution" 
        module, which feeds the data first to a CNN, and then that 
        output to an LSTM. So the data first needs to meet the needs of the CNN, 
        and then the LSTM.
        (timesteps,num_frames,num_features,color_scheme)
        
        Note: Keras adds a dimension to input to represent the "Tensor" that 
        handles the input. This means that sometimes you have to add a 
        shape of (1,) to the shape of the data. 
        
        All this is done here in the generator!
        '''
        self.model_type = model_type
        self.timesteps = timesteps # batch_size appx.
        self.frame_width = frame_width # context_window
        self.batch_size = timesteps * frame_width
        self.samples_per_epoch = data.shape[0]
        self.number_of_batches = self.samples_per_epoch/self.batch_size
        self.counter=0
        self.dict_classes_encountered = {}
        self.data = data

    def generator(self):
        while 1:
            
            #All models, get the features data
            batch = np.array(self.data[self.batch_size*self.counter:self.batch_size*(self.counter+1),]).astype('float32')
            X_batch = batch[:,:-1]
            
            #Only for the LSTM
            if 'lstm' == self.model_type.lower():
                '''
                desired shape to put into model:
                (1,timestep,frame_width,num_features)
                '''
                X_batch = X_batch.reshape((self.timesteps,self.frame_width,X_batch.shape[1]))
                y_batch = batch[:,-1]
                y_indices = list(range(0,len(y_batch),self.frame_width))
                y_batch = y_batch[y_indices]
                
                #keep track of how many different labels are presented in 
                #training, validation, and test datasets
                labels = y_batch[0]
                if labels in self.dict_classes_encountered:
                    self.dict_classes_encountered[labels] += 1
                else:
                    self.dict_classes_encountered[labels] = 1
            
            else:
                #Only for the CNN
                if 'cnn' == self.model_type.lower():
                    #(1,frame_width,num_features,1)
                    X_batch = X_batch.reshape((X_batch.shape[0],X_batch.shape[1],1))
                    
                #Only for the CNN+LSTM
                elif 'cnnlstm' == self.model_type.lower():
                    #(1,timestep,frame_width,num_features,1)
                    #1st 1 --> keras Tensor
                    #2nd 1 --> color scheme
                    X_batch = X_batch.reshape((self.timesteps,self.frame_width,X_batch.shape[1],1))
        
                #Both CNN and CNN+LSTM:
                X_batch = X_batch.reshape((1,)+X_batch.shape)
                y_batch = batch[0,-1]
                y_batch = y_batch.reshape((1,)+y_batch.shape)

                labels = list(set(y_batch))
                if len(labels) > 1:
                    print("Too many labels assigned to one sample")
                if labels[0] in self.dict_classes_encountered:
                    self.dict_classes_encountered[labels[0]] += 1
                else:
                    self.dict_classes_encountered[labels[0]] = 1
            
            #All data:
            #send the batched and reshaped data to models
            self.counter += 1
            yield X_batch,y_batch

            #restart counter to yeild data in the next epoch as well
            if self.counter >= self.number_of_batches:
                self.counter = 0
