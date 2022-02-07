import keras
import tensorflow as tf
from keras import backend as K
import numpy as np

!pip install libmr

import libmr

from keras.datasets import mnist
def get_train_test():
    # batch_size = 128
    num_classes = 10
    # epochs = 50

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print (x_train.shape,y_train.shape)

    # sep_x,sep_y = seperate_data(x_test,y_test)

    # emnist = emnist.read_data_sets('EMNIST_data',one_hot=True)
    # x_train, y_train = emnist.train.images,emnist.train.labels
    # x_test, y_test = emnist.test.images,emnist.test.labels
    # x_valid, y_valid = emnist.validation.images,emnist.validation.labels

    # print (x_train.shape,y_train.shape)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        # x_valid = x_valid.reshape(x_valid.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        # x_valid = x_valid.reshape(x_valid.shape[0],img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)   # noqa: F841

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # x_valid = x_valid.astype('float32')
    x_train /= 255
    x_test /= 255
    # x_valid /= 255
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')
    # print(x_valid.shape[0], 'valid samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test

data = get_train_test()

np.random.seed(12345)

from keras.models import load_model
model = load_model('/content/MNIST_CNN.h5')

#@title
def get_correct_classified(pred, y):
    pred = (pred > 0.5) * 1
    res = np.all(pred == y, axis=1)
    return res

def seperate_data(x, y):
    ind = y.argsort()
    sort_x = x[ind[::-1]]
    sort_y = y[ind[::-1]]

    dataset_x = []
    dataset_y = []
    mark = 0

    for a in range(len(sort_y)-1):
        if sort_y[a] != sort_y[a+1]:
            dataset_x.append(np.array(sort_x[mark:a]))
            dataset_y.append(np.array(sort_y[mark:a]))
            mark = a + 1    # here mark should be updated to the next index.
        if a == len(sort_y)-2:
            dataset_x.append(np.array(sort_x[mark:len(sort_y)]))
            dataset_y.append(np.array(sort_y[mark:len(sort_y)]))
    return dataset_x, dataset_y

def get_activations(model, layer, X_batch):
    # print (model.layers[6].output)
    get_activations = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer].output])
    activations = get_activations([X_batch, 0])[0]
    # print (activations.shape)
    return activations

def compute_feature(x, model):
    score = get_activations(model, 8, x)
    fc8 = get_activations(model, 7, x)
    return score, fc8


def compute_mean_vector(feature):
    return np.mean(feature, axis=0)

import scipy.spatial.distance as spd

def compute_distances(mean_feature, feature, category_name):
    eucos_dist, eu_dist, cos_dist = [], [], []
    eu_dist, cos_dist, eucos_dist = [], [], []
    for feat in feature:
        eu_dist += [spd.euclidean(mean_feature, feat)]
        cos_dist += [spd.cosine(mean_feature, feat)]
        eucos_dist += [spd.euclidean(mean_feature, feat)/200. + spd.cosine(
            mean_feature, feat)]
    distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}
    return distances

#@title
### Weibull functions

def build_weibull(mean, distance, tail):
    weibull_model = {}
    for i in range(len(mean)):
        weibull_model[label[i]] = {}
        weibull = weibull_tailfitting(mean[i], distance[i], tailsize=tail)
        weibull_model[label[i]] = weibull
    return weibull_model


def weibull_tailfitting(
        mean, distance, tailsize=10, distance_type='eucos'):
  
    weibull_model = {}
    # for each category, read meanfile, distance file, and
    # perform weibull fitting
    # for category in labellist:
    # weibull_model = {}
    # distance_scores = loadmat(
    #    '%s/%s_distances.mat' %(distancefiles_path, category))[distance_type]
    # meantrain_vec = loadmat('%s/%s.mat' %(meanfiles_path, category))
    distance_scores = np.array(distance[distance_type])
    meantrain_vec = np.array(mean)

    weibull_model['distances_%s' % distance_type] = distance_scores
    weibull_model['mean_vec'] = meantrain_vec
    weibull_model['weibull_model'] = []
    # for channel in range(NCHANNELS):
    mr = libmr.MR()
    # print (distance_scores.shape)
    tailtofit = sorted(distance_scores)[-tailsize:]
    mr.fit_high(tailtofit, len(tailtofit))
    weibull_model['weibull_model'] += [mr]

    return weibull_model

def query_weibull(
        category_name, weibull_model, distance_type='eucos'):
    """ Query through dictionary for Weibull model.
    Return in the order: [mean_vec, distances, weibull_model]

    Input:
    ------------------------------
    category_name : name of ImageNet category in WNET format. E.g. n01440764
    weibull_model: dictonary of weibull models for
    """
    # print (category_name)
    # print (weibull_model[category_name]['mean_vec'])
    # print (weibull_model[category_name]['distances_%s' %distance_type])
    # print (weibull_model[category_name]['weibull_model'])
    # exit()
    category_weibull = []
    category_weibull += [weibull_model[category_name]['mean_vec']]
    category_weibull += [
        weibull_model[category_name]['distances_%s' % distance_type]]
    category_weibull += [weibull_model[category_name]['weibull_model']]

    return category_weibull

#@title
### compute openmax
NCHANNELS = 1
NCLASSES = 10
ALPHA_RANK = 6
WEIBULL_TAIL_SIZE = 10

def computeOpenMaxProbability(openmax_fc8, openmax_score_u):
    """ Convert the scores in probability value using openmax

    Input:
    ---------------
    openmax_fc8 : modified FC8 layer from Weibull based computation
    openmax_score_u : degree

    Output:
    ---------------
    modified_scores : probability values modified using OpenMax framework,
    by incorporating degree of uncertainity/openness for a given class

    """
    prob_scores, prob_unknowns = [], []
    for channel in range(NCHANNELS):
        channel_scores, channel_unknowns = [], []   # noqa: F841
        for category in range(NCLASSES):
            # print (channel,category)
            # print ('openmax',openmax_fc8[channel, category])

            channel_scores += [sp.exp(openmax_fc8[channel, category])]
        # print ('CS',channel_scores)

        total_denominator = sp.sum(
            sp.exp(openmax_fc8[channel, :])) + sp.exp(
                sp.sum(openmax_score_u[channel, :]))
        # print (total_denominator)

        prob_scores += [channel_scores / total_denominator]
        # print (prob_scores)

        prob_unknowns += [
            sp.exp(sp.sum(openmax_score_u[channel, :]))/total_denominator]

    prob_scores = sp.asarray(prob_scores)
    prob_unknowns = sp.asarray(prob_unknowns)

    scores = sp.mean(prob_scores, axis=0)
    unknowns = sp.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    assert len(modified_scores) == 11
    return modified_scores

# ---------------------------------------------------------------------------------


def recalibrate_scores(weibull_model, labellist, imgarr,
                       layer='fc8', alpharank=6, distance_type='eucos'):
    """
    Given FC8 features for an image, list of weibull models for each class,
    re-calibrate scores

    Input:
    ---------------
    weibull_model : pre-computed weibull_model obtained
     from weibull_tailfitting() function
    labellist : ImageNet 2012 labellist
    imgarr : features for a particular image extracted using caffe architecture

    Output:
    ---------------
    openmax_probab: Probability values for a given class computed using OpenMax
    softmax_probab: Probability values for a given class computed using
     SoftMax (these were precomputed from caffe architecture. Function returns
     them for the sake of convienence)

    """
    imglayer = imgarr[layer]
    ranked_list = imgarr['scores'].argsort().ravel()[::-1]
    alpha_weights = [
        ((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
    ranked_alpha = sp.zeros(10)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # print (imglayer)
    # Now recalibrate each fc8 score for each channel and for each class
    # to include probability of unknown
    openmax_fc8, openmax_score_u = [], []
    for channel in range(NCHANNELS):
        channel_scores = imglayer[channel, :]
        openmax_fc8_channel = []
        openmax_fc8_unknown = []
        # count = 0
        for categoryid in range(NCLASSES):
            # get distance between current channel and mean vector
            category_weibull = query_weibull(
                labellist[categoryid],
                weibull_model, distance_type=distance_type)

            # print (
            #    category_weibull[0], category_weibull[1],category_weibull[2])

            channel_distance = compute_distance(
                channel_scores, channel, category_weibull[0],
                distance_type=distance_type)
            # print ('cd',channel_distance)
            # obtain w_score for the distance and compute probability of the
            # distance
            # being unknown wrt to mean training vector and channel distances
            # for # category and channel under consideration
            wscore = category_weibull[2][channel].w_score(channel_distance)
            # print ('wscore',wscore)
            # print (channel_scores)
            modified_fc8_score = channel_scores[categoryid] * (
                1 - wscore*ranked_alpha[categoryid])
            openmax_fc8_channel += [modified_fc8_score]
            openmax_fc8_unknown += [
                channel_scores[categoryid] - modified_fc8_score]

        # gather modified scores fc8 scores for each channel for the
        # given image
        openmax_fc8 += [openmax_fc8_channel]
        openmax_score_u += [openmax_fc8_unknown]
    openmax_fc8 = sp.asarray(openmax_fc8)
    openmax_score_u = sp.asarray(openmax_score_u)

    # print (openmax_fc8,openmax_score_u)
    # Pass the recalibrated fc8 scores for the image into openmax
    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u)
    softmax_probab = imgarr['scores'].ravel()
    return sp.asarray(openmax_probab), sp.asarray(softmax_probab)

#@title
### openmax.py 
import matplotlib.pyplot as plt
from PIL import Image

def process_input(model, ind, data):
    x_train, x_test, y_train, y_test = data
    imagearr = {}
    plt.imshow(np.squeeze(x_train[ind]))
    plt.show()
    image = np.reshape(x_train[ind], (1, 28, 28, 1))
    score5, fc85 = compute_feature(image, model)
    imagearr['scores'] = score5
    imagearr['fc8'] = fc85
    # print (score5)
    return imagearr

def compute_activation(model, img):
    imagearr = {}
    # img = np.squeeze(img)
    img = np.array(
        Image.fromarray(
            (np.squeeze(img)).astype(np.uint8)).resize((28, 28)))
    # img = scipy.misc.imresize(np.squeeze(img),(28,28))
    # img = img[:,0:28*28]
    img = np.reshape(img, (1, 28, 28, 1))
    score5, fc85 = compute_feature(img, model)
    imagearr['scores'] = score5
    imagearr['fc8'] = fc85
    return imagearr

def image_show(img, labels):
    # print(img.shape)
    # img = scipy.misc.imresize(np.squeeze(img), (28, 28))
    # img = np.array(
    #     Image.fromarray(
    #         (np.squeeze(img)).astype(np.uint8)).resize((28, 28)))
    # print(img.shape)
    # img = img[:, 0:28*28]
    plt.imshow(np.squeeze(img), cmap='gray')
    # print ('Character Label: ',np.argmax(label))
    title = "Original: " + str(
        labels[0]) + " Softmax: " + str(
            labels[1]) + " Openmax: " + str(labels[0])
    plt.title(title, fontsize=8)
    plt.show()

def openmax_known_class(model, y, data):
    x_train, x_test, y_train, y_test = data
    # total = 0
    for i in range(15):
        # print ('label', y[i])
        j = np.random.randint(0, len(y_train[i]))
        imagearr = process_input(model, j)
        print(compute_openmax(model, imagearr))
        #    total += 1
    # print ('correct classified',total,'total set',len(y))

#@title
### openmax_utils.py

def parse_synsetfile(synsetfname):
    """ Read ImageNet 2012 file
    """
    categorylist = open(synsetfname, 'r').readlines()
    imageNetIDs = {}
    count = 0
    for categoryinfo in categorylist:
        wnetid = categoryinfo.split(' ')[0]
        categoryname = ' '.join(categoryinfo.split(' ')[1:])
        imageNetIDs[str(count)] = [wnetid, categoryname]
        count += 1

    assert len(imageNetIDs.keys()) == 1000
    return imageNetIDs


def getlabellist(synsetfname):
    """ read sysnset file as python list. Index corresponds to the output that
    caffe provides
    """

    categorylist = open(synsetfname, 'r').readlines()
    labellist = [category.split(' ')[0] for category in categorylist]
    return labellist

label = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

def create_model(model, data):
    # output = model.layers[-1]

    # Combining the train and test set
    # print (x_train.shape,x_test.shape)
    # exit()
    # x_train, x_test, y_train, y_test = get_train_test()
    x_train, x_test, y_train, y_test = data
    print (x_train.shape,y_train.shape)

    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    pred = model.predict(x_all)
    print(pred)

    index = get_correct_classified(pred, y_all)
    x1_test = x_all[index]
    y1_test = y_all[index]

    y1_test1 = y1_test.argmax(1)

    sep_x, sep_y = seperate_data(x1_test, y1_test1)
    print('hello_1')

    feature = {}
    feature["score"] = []
    feature["fc8"] = []
    weibull_model = {}
    feature_mean = []
    feature_distance = []

    for i in range(len(sep_y)):
        print(i, sep_x[i].shape)
        weibull_model[label[i]] = {}
        score, fc8 = compute_feature(sep_x[i], model)  # 완료
        mean = compute_mean_vector(fc8)   # 완료
        distance = compute_distances(mean, fc8, sep_y)  #완료
        feature_mean.append(mean)
        feature_distance.append(distance)
    np.save('data/mean', feature_mean)
    np.save('data/distance', feature_distance)

create_model(model, data)
