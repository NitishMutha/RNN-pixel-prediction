import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import itertools


def main(isLSTM, RNN_SIZE, RNN_LAYERS, SAVE_DATA, SAMPLES, RESTORE_FOLDER='task2_gru_128', INPAINT_TYPE='1x1'):
    SAVE_PLOTS = True
    SAVE_SAMPLES = True
    PLOTTING = True

    IMAGE_DIM = 1
    TIME_STEP = 783

    EPOCH = 50
    LEARNING_RATE = 0.001  # 0.0003
    KEEP_PROB = .50
    EPSILON = 1e-3
    CLIP = 5.

    SAVE_FOLDER = 'task3_gru_128'

    # INPAINT_TYPE = '2x2' #change as per the type of window - '1x1' or '2x2'

    INPAINT_STEP = 300
    INPAINT_SAMPLES = 1000
    AVG_TRIALS = 10

    np.random.seed(seed=8)

    data_dir = '../inpainting_data/'

    RESTORE_PATH = '../models/' + RESTORE_FOLDER
    SAVE_DATA_PATH = '../Plots/' + SAVE_FOLDER

    def binarize(images, threshold=0.1):
        return (threshold < images).astype(np.float32)

    def readDataSet(type):
        if type == '1x1':
            dataset = np.load(data_dir + 'one_pixel_inpainting.npy')
        else:
            dataset = np.load(data_dir + '2X2_pixels_inpainting.npy')

        images = dataset[0]
        gt_images = dataset[1]

        return images, gt_images

    def fusePixel(img, pixel):
        for i in img:
            i[i == -1] = pixel

        return img

    # fill in the all possible values in the missing pixels
    def getProposedInpaintImages(images_data, type):
        proposed_img = []
        if type == '1x1':
            proposed_img.append(fusePixel(np.copy(images_data), [0]))  # white
            proposed_img.append(fusePixel(np.copy(images_data), [1]))  # black
        else:
            n = 4
            lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
            for l in lst:
                proposed_img.append(fusePixel(np.copy(images_data), l))

        return np.asarray(proposed_img)

    # calculate the min xentropy
    def getMostProbableImages(proposedImgs, xentropy, gt_xentropy):

        min_xentropy_index = np.argmin(xentropy, axis=0)
        min_xentropy = np.min(xentropy, axis=0)

        # proposedImgs
        selectedImages = []
        selectedGTXEnt = []
        for idz, p in enumerate(min_xentropy_index):
            selectedImages.append(np.take(proposedImgs, p, axis=0)[idz])
            selectedGTXEnt.append(np.take(gt_xentropy, p, axis=0)[idz])

        return min_xentropy, selectedImages, selectedGTXEnt

    def plotXentopyGraphs(inpaint_pred, inpaint_gt, title, store_path, save):
        plt.figure(figsize=(23, 6), dpi=50)
        plt.plot(inpaint_pred, 'r-s', label='Ground Truth')
        plt.plot(inpaint_gt, 'b-o', label='Sample')
        plt.ylabel('Cross Entropy', fontsize=18)
        plt.xlabel('Image number', fontsize=18)
        plt.title('Cross Entropy (' + INPAINT_TYPE + '): ' + title, fontsize=22, fontweight='bold')
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(store_path + '/plot-task3-' + INPAINT_TYPE + '.eps', format='eps', dpi=50)
        else:
            plt.show()

    def visualizeMostProbable(GTImages, InPaintImages, mostProbableImage, mostProbableGTXent, mostProbableXnt,
                              store_folder, save, title):

        # draw plots

        fig = plt.figure(figsize=(20, 5), dpi=100)
        fig.suptitle('Most probable in-painting ' + title, fontsize=22, fontweight='bold')
        sub1 = fig.add_subplot(131)
        sub1.set_xlabel('Original Image', fontsize=18)
        sub1.imshow(np.reshape(GTImages, (28, 28)), interpolation='None', vmin=-1, vmax=1)

        sub2 = fig.add_subplot(132)
        sub2.set_xlabel('Missing pixel', fontsize=18)
        sub2.imshow(np.reshape(InPaintImages, (28, 28)), interpolation='None', vmin=-1, vmax=1)

        sub3 = fig.add_subplot(133)
        sub3.set_title('GT: %.2f, Sample: %.2f' % (mostProbableGTXent, mostProbableXnt), fontsize=16)
        sub3.set_xlabel('In-painting', fontsize=18)
        sub3.imshow(np.reshape(mostProbableImage, (28, 28)), interpolation='None', vmin=-1, vmax=1)
        fig.tight_layout()
        if save:
            fig.savefig(store_folder + '/plot-tsak3b' + title + '.eps', format='eps', dpi=100)
        else:
            fig.show()

    # code start


    if isLSTM:
        modelType = 'lstm'
    else:
        modelType = 'gru'

    x = tf.placeholder(tf.float32, [None, TIME_STEP, IMAGE_DIM])
    y = tf.placeholder(tf.float32, [None, TIME_STEP])

    def RNN(x, model=isLSTM, layers=1, name="rnn"):
        with tf.name_scope(name):
            if model:
                cell = rnn.LSTMCell(RNN_SIZE, state_is_tuple=True, forget_bias=1.0)
            else:
                cell = rnn.GRUCell(RNN_SIZE)

            cell = rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=0.9)

            if layers > 1:
                cell = rnn.MultiRNNCell([cell] * layers)

            output, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
            output_ustacked = tf.unstack(output, axis=1)
            predict = tf.contrib.layers.fully_connected(inputs=output_ustacked, num_outputs=1, activation_fn=None)

            predict = tf.squeeze(predict, axis=2)
            predict = tf.transpose(predict)

            # batch normalization
            batch_mean, batch_var = tf.nn.moments(predict, [0])
            scale = tf.Variable(tf.ones([TIME_STEP]))
            beta = tf.Variable(tf.zeros([TIME_STEP]))
            predict = tf.nn.batch_normalization(predict, batch_mean, batch_var, beta, scale, EPSILON)

            return predict

    prediction = RNN(x, isLSTM, RNN_LAYERS, "rnn")

    with tf.name_scope("Loss"):
        # Softmax loss and L2
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
        tf.summary.scalar("cross_entropy", cost)

    with tf.name_scope("train"):
        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        gvs = train_op.compute_gradients(cost)
        capped_gvs = [(tf.clip_by_value(grad, -CLIP, CLIP), var) for grad, var in gvs if grad is not None]
        optimizer = train_op.apply_gradients(capped_gvs)

    with tf.name_scope("accuracy"):
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)

        saver.restore(sess, RESTORE_PATH + '/model.ckpt')

        InPaintImages, GTImages = readDataSet(INPAINT_TYPE)

        allProposedImages = getProposedInpaintImages(np.copy(InPaintImages), INPAINT_TYPE)

        inpaint_gt_loss = np.array([])
        inpaint_predict_loss = np.array([])
        inpaint_predict_img = []

        n_proposals = allProposedImages.shape[0]

        for proposal in range(n_proposals):
            proposedImage = np.copy(allProposedImages[proposal])

            batch_x = proposedImage[:, 0:TIME_STEP]
            batch_y = proposedImage[:, 1:TIME_STEP + 1]

            batch_x = batch_x.reshape(INPAINT_SAMPLES, TIME_STEP, IMAGE_DIM)

            # Get the prediction of the next pixels in the image from the model
            predicted_img = prediction.eval(feed_dict={x: batch_x, y: batch_y})
            proposed_img_prob = tf.nn.sigmoid(predicted_img).eval()

            # calculate the cross entropy for ground truth
            gt_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=GTImages[:, 1:784],
                                                                               logits=proposed_img_prob),
                                       axis=1)
            # calculate the cross entropy for predicted sample
            predict_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=proposedImage[:, 1:784],
                                                                                    logits=proposed_img_prob),
                                            axis=1)
            inpaint_gt_loss = np.append(inpaint_gt_loss, np.array(gt_entropy.eval()))
            inpaint_predict_loss = np.append(inpaint_predict_loss,
                                             np.array(predict_entropy.eval()))
            proposedImage = np.insert(proposedImage[:, 1:784], 0, 0, axis=1)
            inpaint_predict_img.append(proposedImage)

        # construct final list of all images with xentropy and image data
        inpaint_gt_loss = np.reshape(inpaint_gt_loss, (n_proposals, INPAINT_SAMPLES))
        inpaint_predict_loss = np.reshape(inpaint_predict_loss, (n_proposals, INPAINT_SAMPLES))
        inpaint_predict_img = np.asarray(inpaint_predict_img)

        mostProbableXnt, mostProbableImage, mostProbableGTXent = getMostProbableImages(inpaint_predict_img,
                                                                                       inpaint_predict_loss,
                                                                                       inpaint_gt_loss)

        # calculate the accuracy of the prediction of the missing pixel
        acc = []
        for j in range(GTImages.shape[0]):
            acc.append(accuracy_score(GTImages[j], mostProbableImage[j]))

        print('accuracy: ' + str(np.mean(acc)))

        # plot the cross entropy comparision with GT
        if PLOTTING:

            plotXentopyGraphs(mostProbableXnt, mostProbableGTXent,
                              'Comparision of most probable cross entropy with ground truth', SAVE_DATA_PATH,
                              SAVE_PLOTS)

            for v in SAMPLES:
                vv = int(v)
                visualizeMostProbable(GTImages[vv], InPaintImages[vv], mostProbableImage[vv], mostProbableGTXent[vv],
                                      mostProbableXnt[vv], SAVE_DATA_PATH, SAVE_PLOTS, INPAINT_TYPE + '_' + str(v), )

        # save the data in to the npy file
        if SAVE_DATA:
            np.savez(data_dir + '/' + INPAINT_TYPE + '_most-probable-images-output-file',
                     image_data=InPaintImages,
                     gt_image=GTImages,
                     most_probable_img=mostProbableImage,
                     most_probable_xent=mostProbableXnt,
                     most_probable_gt_xent=mostProbableGTXent)

            if INPAINT_TYPE == '1x1':
                sfile = data_dir + 'one_pixel_inpainting_output.npy'
            else:
                sfile = data_dir + '2X2_pixels_inpainting_output.npy'
            np.save(sfile, [InPaintImages, GTImages, mostProbableImage])

            #save to csv
            df = pd.DataFrame({'xentGT': mostProbableGTXent,
                               'xentSAM': mostProbableXnt
                               })
            df.to_csv(SAVE_DATA_PATH + "/xentropy_gru_" + INPAINT_TYPE + str(RNN_SIZE) + '_' + str(RNN_LAYERS) + ".csv",
                      index=False)

        print('finish')


# in order to run code from console command: python task2b.py <arg1> <arg2> <arg3> <arg4> <arg5> <arg6>
# arg1 = 1x1 / 2x2
# arg2 = lstm / gru
# arg3 = (int) rnn size e.g. 32 or 64 or 128
# arg 4 = (int) number of layers, eg, 1 or 3
# arg 5 = save / nosave
# arg 6 = (comma separate value) sample numbers to run. eg. 1,2,3,4
if __name__ == '__main__':
    args = sys.argv

    if len(args) > 1:
        INPAINT_TYPE = args[1]
        isLSTM = args[2].lower() == 'lstm'
        RNN_SIZE = args[3]
        RNN_LAYERS = int(args[4])
        SAVE_DATA = args[5].lower() == 'save'
        SAMPLES = str(args[6]).split(',')
    else:
        INPAINT_TYPE = '1x1'
        isLSTM = False
        RNN_SIZE = '128'
        RNN_LAYERS = 1
        SAVE_DATA = True
        SAMPLES = [9, 10, 11, 12, 13, 0]

    FOLDER = 'task2_'
    if isLSTM and RNN_LAYERS == 1:
        FOLDER += 'lstm_' + RNN_SIZE
    elif isLSTM and RNN_LAYERS > 1:
        FOLDER += 'lstm_' + RNN_SIZE + '_' + str(RNN_LAYERS)
    elif not isLSTM and RNN_LAYERS == 1:
        FOLDER += 'gru_' + RNN_SIZE
    else:
        FOLDER += 'gru_' + RNN_SIZE + '_' + str(RNN_LAYERS)

    main(isLSTM, int(RNN_SIZE), RNN_LAYERS, SAVE_DATA, SAMPLES, FOLDER, INPAINT_TYPE)
