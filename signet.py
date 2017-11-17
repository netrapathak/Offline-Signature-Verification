
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os

#from keras.utils.visualize_util import plot

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras import backend as K
import sys
import glob
import logging
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import random
random.seed(1337)

# Create a session for running Ops on the Graph.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger()

def euclidean_distance(vects):
    assert len(vects) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
    # return (shape1[1], 0)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''

    # y_pred --> Dw, euclidean distance computed in the embedded space

    """ What you want is double the distance if the pairs are equal - 
    this is the loss for pairs should be be with "zero" distance. 
    But if the pairs are distinct from each other you want to calculate
    their distance from the margin and double it.
    """

    margin = 1

    """ Dw to be close to 0 when y_true is 1 (for positive pairs) and 
        Dw close or bigger than 1 when y_true is 0 (for negative pairs)
        Distance low for similar pairs and high for diff images
    """    
    
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    
    """In the case below, similar pairs: y=0 and dissimilar y=1
       Dw = 1 for similar pairs 
    """
    # return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
    # corresponding label += [0,1] instead of label += [1,0]


def create_base_network_signet(input_shape):
    
    seq = Sequential()
    seq.add(Convolution2D(96, (11, 11), activation='relu', name='conv1_1', strides=(4,4), input_shape= input_shape, 
                        kernel_initializer='glorot_uniform', data_format='channels_last'))
    seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))    
    seq.add(ZeroPadding2D((2, 2),))
    
    seq.add(Convolution2D(256, (5, 5), activation='relu', name='conv2_1', strides=(1, 1), kernel_initializer='glorot_uniform'))
    seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
    seq.add(ZeroPadding2D((1, 1)))
    
    seq.add(Convolution2D(384, (3, 3), activation='relu', name='conv3_1', strides=(1, 1), kernel_initializer='glorot_uniform'))
    seq.add(ZeroPadding2D((1, 1)))
    
    seq.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_2', strides=(1, 1), kernel_initializer='glorot_uniform'))    
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
#    model.add(SpatialPyramidPooling([1, 2, 4]))
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(1024, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
    seq.add(Dropout(0.5))
    
    seq.add(Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform')) # softmax changed to relu
    
    print (seq.summary())
    return seq

def compute_accuracy_roc(predictions, labels):
   # Compute ROC accuracy with a range of thresholds on distances.
   dmax = np.max(predictions)
   dmin = np.min(predictions)
   nsame = np.sum(labels == 1)
   ndiff = np.sum(labels == 0)
   
   step = 0.01
   max_acc = 0
   
   for d in np.arange(dmin, dmax+step, step):
       idx1 = predictions.ravel() <= d
       idx2 = predictions.ravel() > d
       
       tpr = float(np.sum(labels[idx1] == 1)) / nsame       
       tnr = float(np.sum(labels[idx2] == 0)) / ndiff
       acc = 0.5 * (tpr + tnr)       
#       print ('ROC', acc, tpr, tnr)
       
       if (acc > max_acc):
           max_acc = acc
           
   return max_acc

def compute_accuracy(predictions, labels):
    """ Compute classification accuracy with a fixed threshold on distances.
    """
    return labels[predictions.ravel() < 0.5].mean()
    # return np.mean(labels==(predictions.ravel() > 0.5))

def create_data(height,width,di):
    files = glob.glob(dir+'/*.png')
    nb_classes = 55 #cedar dataset
    # filename from path: file.rsplit('/', 1)[-1]
    # cedar data specific, will get 1_1; original from start and .png from end removed
    img_names = [file.rsplit('/', 1)[-1][9:-4] for file in files]
    data = []
    labels = []
    for i in xrange(len(img_names)):
        class_index = int(img_names[i].split('_',1)[0])
        label = class_index - 1
        img = image.load_img(files[i], grayscale = True, target_size=(height, width))
        img = image.img_to_array(img)#, dim_ordering='tf')
        data.append(img)
        labels.append(label)

    return np.array(data), np.array(labels), nb_classes
 
def create_pairs(x, digit_indices, nb_classes):
    """      x:         X_train, array of array of all train samples.
        digit_indices:  List of array, length = no of classes; each sublist consists of train sample indices
                        belonging to that particular class index/class
    """
    """ Positive and negative pair creation.
        Alternates between positive and negative pairs.
    """
    pairs = []
    labels = []
    
    print ('\n\n')
    print ('X_train shape:  ', x.shape)
    print ('Digit_indices shape:    ', np.array(digit_indices).shape)
    print ('Length of digit indices:    ', len(digit_indices))
    print ('No of classes:  ', nb_classes)
    print ('\n\n')

    n = min([len(digit_indices[d]) for d in range(nb_classes)]) - 1
    for d in range(nb_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, nb_classes)
            dn = (d + inc) % nb_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]    #check label based on similarity = 0 or 1
            # labels += [0,1]     #similar pairs = 0 in this case
    return np.array(pairs), np.array(labels)

def format_input(X):
    if not(type(X) is np.ndarray):   
        X = np.array(X)
        print ('Formatted input by converting to array')
    try:
        print ('Input shape before reshaping:   ', X.shape)
        # X = X.reshape(X.shape[0], X.shape[1], img_height, img_width, 1)
        X = X.reshape(X.shape[0], img_height, img_width, 1)
        X = X.astype('float32')
        X /= 255

        return X
    except Exception as e:
        print ('EXCEPTION while reshaping and formatting input data X:  \n', e)
        print (log.exception('ERROR MESSAGE'))

# parameters
img_height = 155
img_width = 220
featurewise_std_normalization = True
epochs = 20
batch_size = 128    
input_shape=(img_height, img_width, 1)

cedar_data_dir = '../SiameseData/signatures_cedar/full_org'
X_train, y_train, nb_classes = create_data(img_height, img_width, cedar_data_dir)    
X_train = format_input(X_train)

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(nb_classes)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices, nb_classes)

for d in xrange(len(digit_indices)):    
        # 24 instances of each class in cedar data    
        print('No. of instances for class %d:   %d'%(d,len(list(digit_indices[d]))))

# X_test = format_input(X_test)
# y_test = 
# digit_indices = [np.where(y_test == i)[0] for i in range(nb_classes)]
# te_pairs, te_y = create_pairs(X_test, digit_indices)
    
# network definition
base_network = create_base_network_signet(input_shape)

input_a = Input(shape=(input_shape))
input_b = Input(shape=(input_shape))

# because we re-use the same instance `base_network`,
# the weights of the network will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(inputs=[input_a, input_b], outputs=distance)
    
# compile model
rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
adadelta = Adadelta()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=batch_size, epochs=epochs)
model.save('../SiameseData/Signet_Models/cedar_original/model.h5')
# display model
# plot(model, show_shapes=True)
# sys.exit()     
    
# # callbacks
# fname = 'model_weights.hdf5'

# checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
# tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_images=True)

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
# tr_acc = compute_accuracy_roc(pred, tr_y)
tr_acc = compute_accuracy(pred, tr_y)
# pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
# te_acc = compute_accuracy(pred, te_y)



# # model.fit_generator(generator=datagen.next_train(), samples_per_epoch=datagen.samples_per_train, nb_epoch=nb_epoch,
# #                     validation_data=datagen.next_valid(), nb_val_samples=int(datagen.samples_per_valid))   # KERAS 1

# model.fit_generator(generator=datagen.next_train(), steps_per_epoch=960, epochs=nb_epoch,
# #                     validation_data=datagen.next_valid(), validation_steps=120, callbacks=[checkpointer])  # KERAS 2
    

# # tr_pred = model.predict_generator(generator=datagen.next_train(), val_samples=int(datagen.samples_per_train))
# te_pred = model.predict_generator(generator=datagen.next_test(), steps=120)

# #tr_acc = compute_accuracy_roc(tr_pred, datagen.train_labels)
# te_acc = compute_accuracy_roc(te_pred, datagen.test_labels)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
# print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
