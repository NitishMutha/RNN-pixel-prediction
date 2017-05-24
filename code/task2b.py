import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import pandas as pd

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

INPAINT_STEP = 300
INPAINT_SAMPLES = 100
AVG_TRIALS = 10


def binarize(images, threshold=0.1):
    return (threshold < images).astype('float')


def processData(images):
    for im in range(len(images)):
        images[im] = binarize(images[im])

    return images


def createInPaintDataSet(rawImages):
    GTimg = rawImages[np.random.choice(rawImages.shape[0], INPAINT_SAMPLES, replace=False), :]
    InPaintIm = np.copy(GTimg)

    InPaintIm[:, 484:] = 0.

    return InPaintIm, GTimg


def computeMean(x):
    return np.mean(x, axis=0)


def computeVariance(x):
    return np.std(x, axis=0)


def visualize1s(inpaints, img_id, title, store_folder, save):
    gt_img = inpaints['gt_img'][img_id]
    pred_img = inpaints['predict_img'][img_id]

    x_gt = inpaints['x_gt'][img_id]
    x_sample = inpaints['x_sample'][img_id]

    # draw plots
    imgs = [gt_img, pred_img]
    fig = plt.figure(figsize=(20, 5), dpi=50)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for p in range(len(imgs)):
        sub1 = fig.add_subplot(int('1' + str(len(imgs)) + str(p + 1)))
        if p == 0:
            sub1.set_xlabel('Original Image', fontsize=12)
        else:
            sub1.set_title('GT: %.2f, Sample: %.2f' % (x_gt, x_sample))
            sub1.set_xlabel('Sample Image ' + str(p + 1), fontsize=12)
        sub1.imshow(np.reshape(imgs[p], (28, 28)))

    fig.tight_layout()
    if save:
        fig.savefig(store_folder + '/' + title + '.eps', format='eps', dpi=50)
    else:
        fig.show()


def visualize(inpaints, img_id, title, store_folder, save):
    gt_img = inpaints['gt_img'][img_id]
    pred_img = inpaints['predict_img'][img_id][:5]

    x_gt = inpaints['x_gt'][img_id][:5]
    x_sample = inpaints['x_sample'][img_id][:5]

    # draw plots
    imgs = np.squeeze(np.insert(pred_img, 0, gt_img, axis=0))
    fig = plt.figure(figsize=(20, 5), dpi=50)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for p in range(imgs.shape[0]):
        sub1 = fig.add_subplot(int('1' + str(int(imgs.shape[0])) + str(p + 1)))
        if p == 0:
            sub1.set_xlabel('Original Image', fontsize=12)
        else:
            sub1.set_title('GT: %.2f, Sample: %.2f' % (x_gt[p - 1], x_sample[p - 1]))
            sub1.set_xlabel('Sample Image ' + str(p + 1), fontsize=12)
        sub1.imshow(np.reshape(imgs[p], (28, 28)))

    fig.tight_layout()

    if save:
        fig.savefig(store_folder + '/' + title + '.eps', format='eps', dpi=50)
    else:
        fig.show()


def visualizeSamples(samples, inpaint_1_step_avg_loss, inpaint_10_steps_avg_loss, inpaint_28_steps_avg_loss,
                     inpaint_300_steps_avg_loss, store_folder, save):
    for s in samples:
        ss = int(s)
        # 1 step
        # visualize(inpaint_1_step_avg_loss, s)
        visualize1s(inpaint_1_step_avg_loss, ss, '1 Step Pixel Prediction - Image no ' + str(s), store_folder, save)

        # Select 3 examples and visualize
        # 10 Steps
        visualize(inpaint_10_steps_avg_loss, ss, '10 Step Pixel Prediction - Image no ' + str(s), store_folder, save)

        # 28 Steps
        visualize(inpaint_28_steps_avg_loss, ss, '28 Step Pixel Prediction - Image no ' + str(s), store_folder, save)

        # 300 Steps
        visualize(inpaint_300_steps_avg_loss, ss, '300 Step Pixel Prediction - Image no ' + str(s), store_folder, save)


def plotXentopyGraphs(inpaint, title, store_path, save):
    plt.figure(figsize=(20, 6), dpi=50)
    plt.plot(inpaint['mean_gt'], 'r-s', label='Ground Truth')
    plt.plot(inpaint['mean_sample'], 'b-o', label='Sample')
    plt.ylabel('Cross Entropy', fontsize=18)
    plt.xlabel('Image number', fontsize=18)
    plt.title('Cross Entropy: ' + title, fontsize=22, fontweight='bold')
    plt.grid(True)
    plt.legend()
    if save:
        plt.savefig(store_path + '/plot-' + title + '.eps', format='eps', dpi=100)
    else:
        plt.show()


# Run the saved model and get the plots and visualisation
def runModel(SAVE_DATA_PATH, samples, store_folder):
    inpaint_1_step_avg_loss = {}
    inpaint_10_steps_avg_loss = {}
    inpaint_28_steps_avg_loss = {}
    inpaint_300_steps_avg_loss = {}

    # Load the saved model form the saved file.
    data_1 = np.load(SAVE_DATA_PATH + '/inpaint_1_' + str(RNN_SIZE) + '_' + str(RNN_LAYERS) + '.npz')
    inpaint_1_step_avg_loss['x_gt'] = data_1['x_gt_1']
    inpaint_1_step_avg_loss['mean_gt'] = data_1['mean_gt_1']
    inpaint_1_step_avg_loss['variance_gt'] = data_1['variance_gt_1']
    inpaint_1_step_avg_loss['x_sample'] = data_1['x_sample_1']
    inpaint_1_step_avg_loss['mean_sample'] = data_1['mean_sample_1']
    inpaint_1_step_avg_loss['variance_sample'] = data_1['variance_sample_1']
    inpaint_1_step_avg_loss['predict_img'] = data_1['predict_img_1']
    inpaint_1_step_avg_loss['gt_img'] = data_1['gt_img_1']

    data_10 = np.load(SAVE_DATA_PATH + '/inpaint_10_' + str(RNN_SIZE) + '_' + str(RNN_LAYERS) + '.npz')
    inpaint_10_steps_avg_loss['x_gt'] = data_10['x_gt_10']
    inpaint_10_steps_avg_loss['mean_gt'] = data_10['mean_gt_10']
    inpaint_10_steps_avg_loss['variance_gt'] = data_10['variance_gt_10']
    inpaint_10_steps_avg_loss['x_sample'] = data_10['x_sample_10']
    inpaint_10_steps_avg_loss['mean_sample'] = data_10['mean_sample_10']
    inpaint_10_steps_avg_loss['variance_sample'] = data_10['variance_sample_10']
    inpaint_10_steps_avg_loss['predict_img'] = data_10['predict_img_10']
    inpaint_10_steps_avg_loss['gt_img'] = data_10['gt_img_10']

    data_28 = np.load(SAVE_DATA_PATH + '/inpaint_28_' + str(RNN_SIZE) + '_' + str(RNN_LAYERS) + '.npz')
    inpaint_28_steps_avg_loss['x_gt'] = data_28['x_gt_28']
    inpaint_28_steps_avg_loss['mean_gt'] = data_28['mean_gt_28']
    inpaint_28_steps_avg_loss['variance_gt'] = data_28['variance_gt_28']
    inpaint_28_steps_avg_loss['x_sample'] = data_28['x_sample_28']
    inpaint_28_steps_avg_loss['mean_sample'] = data_28['mean_sample_28']
    inpaint_28_steps_avg_loss['variance_sample'] = data_28['variance_sample_28']
    inpaint_28_steps_avg_loss['predict_img'] = data_28['predict_img_28']
    inpaint_28_steps_avg_loss['gt_img'] = data_28['gt_img_28']

    data_300 = np.load(SAVE_DATA_PATH + '/inpaint_300_' + str(RNN_SIZE) + '_' + str(RNN_LAYERS) + '.npz')
    inpaint_300_steps_avg_loss['x_gt'] = data_300['x_gt_300']
    inpaint_300_steps_avg_loss['mean_gt'] = data_300['mean_gt_300']
    inpaint_300_steps_avg_loss['variance_gt'] = data_300['variance_gt_300']
    inpaint_300_steps_avg_loss['x_sample'] = data_300['x_sample_300']
    inpaint_300_steps_avg_loss['mean_sample'] = data_300['mean_sample_300']
    inpaint_300_steps_avg_loss['variance_sample'] = data_300['variance_sample_300']
    inpaint_300_steps_avg_loss['predict_img'] = data_300['predict_img_300']
    inpaint_300_steps_avg_loss['gt_img'] = data_300['gt_img_300']

    visualizeSamples(samples, inpaint_1_step_avg_loss, inpaint_10_steps_avg_loss, inpaint_28_steps_avg_loss,
                     inpaint_300_steps_avg_loss, store_folder, SAVE_SAMPLES)

    if PLOTTING:
        plotXentopyGraphs(inpaint_1_step_avg_loss, '1 Step prediction on 100 images', store_folder, SAVE_PLOTS)
        plotXentopyGraphs(inpaint_10_steps_avg_loss, 'Averaged over 10 samples of 10 Steps prediction on 100 images',
                          store_folder, SAVE_PLOTS)
        plotXentopyGraphs(inpaint_28_steps_avg_loss, 'Averaged over 10 samples of 28 Steps prediction on 100 images',
                          store_folder, SAVE_PLOTS)
        plotXentopyGraphs(inpaint_300_steps_avg_loss, 'Averaged over 10 samples of 300 Steps prediction on 100 images',
                          store_folder, SAVE_PLOTS)


def main(MODE=True, isLSTM=False, RNN_SIZE=128, RNN_LAYERS=1, RESTORE_FOLDER='task2_gru_128', SAVE_DATA=True,
         SAMPLES=[0, 1, 2]):
    data_dir = '../data/'

    from tensorflow.examples.tutorials.mnist import input_data

    # import dataset with one-hot class encoding
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    RESTORE_PATH = '../models/' + RESTORE_FOLDER
    SAVE_DATA_PATH = '../Plots/' + RESTORE_FOLDER

    np.random.seed(seed=8)

    # code start

    # select mode to build model or just run the pre saved model
    if MODE:
        # Running in build mode
        print('Running in build mode..')

        processData(mnist.test.images)

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

            InPaintImages, GTImages = createInPaintDataSet(mnist.test.images)

            inpaint_1_step_avg_loss = {}
            inpaint_10_steps_avg_loss = {}
            inpaint_28_steps_avg_loss = {}
            inpaint_300_steps_avg_loss = {}

            inpaint_1_step_gt_loss = np.array([])
            inpaint_10_steps_gt_loss = np.array([])
            inpaint_28_steps_gt_loss = np.array([])
            inpaint_300_steps_gt_loss = np.array([])

            inpaint_1_step_sample_loss = np.array([])
            inpaint_10_steps_sample_loss = np.array([])
            inpaint_28_steps_sample_loss = np.array([])
            inpaint_300_steps_sample_loss = np.array([])

            predict_img_1 = []
            predict_img_10 = []
            predict_img_28 = []
            predict_img_300 = []

            # iterated through the same image to average the cross entropy for multi step
            for avg_itr in range(AVG_TRIALS):

                PIXEL_LOC = 484
                inpaint_1_step_prob = []  # dim = 1 x 1
                inpaint_10_steps_prob = []  # dim = 10 x 1
                inpaint_28_steps_prob = []  # dim = 28 x 1
                inpaint_300_steps_prob = []  # dim = 300 x 1

                inpaint_1_step_bin = []  # dim = 1 x 1
                inpaint_10_steps_bin = []  # dim = 10 x 1
                inpaint_28_steps_bin = []  # dim = 28 x 1
                inpaint_300_steps_bin = []  # dim = 300 x 1

                gt_image = np.copy(GTImages)
                next_in_impaint_image = np.copy(InPaintImages)

                gt_1 = gt_image[:, :PIXEL_LOC + 1]
                gt_10 = gt_image[:, :PIXEL_LOC + 10]
                gt_28 = gt_image[:, :PIXEL_LOC + 28]
                gt_300 = gt_image[:, 1:PIXEL_LOC + 300]

                # interate through the same image to predict the next pixel for remaining timesteps
                for itr in range(INPAINT_STEP):

                    batch_x = next_in_impaint_image[:, 0:TIME_STEP]
                    batch_y = next_in_impaint_image[:, 1:TIME_STEP + 1]

                    batch_x = batch_x.reshape((INPAINT_SAMPLES, TIME_STEP, IMAGE_DIM))

                    predicted_img = prediction.eval(feed_dict={x: batch_x, y: batch_y})
                    next_predicted_pixel_prob = tf.nn.sigmoid(predicted_img[:, PIXEL_LOC - 1:PIXEL_LOC]).eval()
                    pred_prob = tf.nn.sigmoid(predicted_img).eval()
                    print("Iteration: " + str(avg_itr + 1) + " | Pixel: " + str(itr + 1))

                    # next_predicted_pixel_binary = binarize(next_predicted_pixel_prob, np.random.uniform(0.0,0.3))
                    next_predicted_pixel_binary = binarize(next_predicted_pixel_prob, 0.25)

                    # assign to next pixel location
                    next_in_impaint_image[:, PIXEL_LOC:PIXEL_LOC + 1] = next_predicted_pixel_binary

                    if itr == 0 and avg_itr < 1:
                        # for 1 step just have one record.
                        inpaint_1_step_prob = np.squeeze(pred_prob[:, :PIXEL_LOC + 1])
                        inpaint_1_step_bin = np.squeeze(next_in_impaint_image[:, :PIXEL_LOC + 1])
                        if itr == 0:
                            predict_img_1 = np.copy(next_in_impaint_image)

                    if itr == 9:
                        inpaint_10_steps_prob = np.squeeze(pred_prob[:, :PIXEL_LOC + 1])
                        inpaint_10_steps_bin = np.squeeze(next_in_impaint_image[:, :PIXEL_LOC + 1])
                        predict_img_10.append(np.copy(next_in_impaint_image))

                    if itr == 27:
                        inpaint_28_steps_prob = np.squeeze(pred_prob[:, :PIXEL_LOC + 1])
                        inpaint_28_steps_bin = np.squeeze(next_in_impaint_image[:, :PIXEL_LOC + 1])
                        predict_img_28.append(np.copy(next_in_impaint_image))

                    if itr == 299:
                        inpaint_300_steps_prob = np.squeeze(pred_prob[:, :PIXEL_LOC + 1])
                        inpaint_300_steps_bin = np.squeeze(next_in_impaint_image[:, :PIXEL_LOC])
                        predict_img_300.append(np.copy(next_in_impaint_image))

                    PIXEL_LOC += 1

                # End of image in painting

                # 1 Step xentropy
                if avg_itr < 1:
                    gt_entropy_1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_1,
                                                                                         logits=inpaint_1_step_prob),
                                                 axis=1)
                    predict_entropy_1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inpaint_1_step_bin,
                                                                                              logits=inpaint_1_step_prob),
                                                      axis=1)
                    inpaint_1_step_gt_loss = np.append(inpaint_1_step_gt_loss, np.array(gt_entropy_1.eval()))
                    inpaint_1_step_sample_loss = np.append(inpaint_1_step_sample_loss,
                                                           np.array(predict_entropy_1.eval()))

                # 10 Steps xentropy
                gt_entropy_10 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_10,
                                                                                      logits=inpaint_10_steps_prob),
                                              axis=1)
                predict_entropy_10 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inpaint_10_steps_bin,
                                                                                           logits=inpaint_10_steps_prob),
                                                   axis=1)

                inpaint_10_steps_gt_loss = np.append(inpaint_10_steps_gt_loss, np.array(gt_entropy_10.eval()))
                inpaint_10_steps_sample_loss = np.append(inpaint_10_steps_sample_loss,
                                                         np.array(predict_entropy_10.eval()))

                # 28 Steps xentropy
                gt_entropy_28 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_28,
                                                                                      logits=inpaint_28_steps_prob),
                                              axis=1)
                predict_entropy_28 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inpaint_28_steps_bin,
                                                                                           logits=inpaint_28_steps_prob),
                                                   axis=1)

                inpaint_28_steps_gt_loss = np.append(inpaint_28_steps_gt_loss, np.array(gt_entropy_28.eval()))
                inpaint_28_steps_sample_loss = np.append(inpaint_28_steps_sample_loss,
                                                         np.array(predict_entropy_28.eval()))

                # 300 Steps xentropy
                gt_entropy_300 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_300,
                                                                                       logits=inpaint_300_steps_prob),
                                               axis=1)
                predict_entropy_300 = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=inpaint_300_steps_bin,
                                                            logits=inpaint_300_steps_prob),
                    axis=1)
                inpaint_300_steps_gt_loss = np.append(inpaint_300_steps_gt_loss, np.array(gt_entropy_300.eval()))
                inpaint_300_steps_sample_loss = np.append(inpaint_300_steps_sample_loss,
                                                          np.array(predict_entropy_300.eval()))

            # End of 10 trials on same image

            # average the losses for each image
            inpaint_1_step_avg_loss['x_gt'] = inpaint_1_step_gt_loss
            inpaint_1_step_avg_loss['mean_gt'] = inpaint_1_step_gt_loss
            inpaint_1_step_avg_loss['variance_gt'] = inpaint_1_step_gt_loss
            inpaint_1_step_avg_loss['x_sample'] = inpaint_1_step_sample_loss
            inpaint_1_step_avg_loss['mean_sample'] = inpaint_1_step_sample_loss
            inpaint_1_step_avg_loss['variance_sample'] = inpaint_1_step_sample_loss
            inpaint_1_step_avg_loss['predict_img'] = predict_img_1
            inpaint_1_step_avg_loss['gt_img'] = GTImages

            # 10
            inpaint_10_steps_gt_loss = np.reshape(inpaint_10_steps_gt_loss, (AVG_TRIALS, INPAINT_SAMPLES))
            inpaint_10_steps_sample_loss = np.reshape(inpaint_10_steps_sample_loss, (AVG_TRIALS, INPAINT_SAMPLES))

            inpaint_10_steps_avg_loss['x_gt'] = inpaint_10_steps_gt_loss.T
            inpaint_10_steps_avg_loss['mean_gt'] = computeMean(inpaint_10_steps_gt_loss)
            inpaint_10_steps_avg_loss['variance_gt'] = computeVariance(inpaint_10_steps_gt_loss)

            inpaint_10_steps_avg_loss['x_sample'] = inpaint_10_steps_sample_loss.T
            inpaint_10_steps_avg_loss['mean_sample'] = computeMean(inpaint_10_steps_sample_loss)
            inpaint_10_steps_avg_loss['variance_sample'] = computeVariance(inpaint_10_steps_sample_loss)

            inpaint_10_steps_avg_loss['predict_img'] = np.transpose(predict_img_10, (1, 0, 2))
            inpaint_10_steps_avg_loss['gt_img'] = GTImages

            # 28
            inpaint_28_steps_gt_loss = np.reshape(inpaint_28_steps_gt_loss, (AVG_TRIALS, INPAINT_SAMPLES))
            inpaint_28_steps_sample_loss = np.reshape(inpaint_28_steps_sample_loss, (AVG_TRIALS, INPAINT_SAMPLES))

            inpaint_28_steps_avg_loss['x_gt'] = inpaint_28_steps_gt_loss.T
            inpaint_28_steps_avg_loss['mean_gt'] = computeMean(inpaint_28_steps_gt_loss)
            inpaint_28_steps_avg_loss['variance_gt'] = computeVariance(inpaint_28_steps_gt_loss)

            inpaint_28_steps_avg_loss['x_sample'] = inpaint_28_steps_sample_loss.T
            inpaint_28_steps_avg_loss['mean_sample'] = computeMean(inpaint_28_steps_sample_loss)
            inpaint_28_steps_avg_loss['variance_sample'] = computeVariance(inpaint_28_steps_sample_loss)

            inpaint_28_steps_avg_loss['predict_img'] = np.transpose(predict_img_28, (1, 0, 2))
            inpaint_28_steps_avg_loss['gt_img'] = GTImages

            # 300
            inpaint_300_steps_gt_loss = np.reshape(inpaint_300_steps_gt_loss, (AVG_TRIALS, INPAINT_SAMPLES))
            inpaint_300_steps_sample_loss = np.reshape(inpaint_300_steps_sample_loss, (AVG_TRIALS, INPAINT_SAMPLES))

            inpaint_300_steps_avg_loss['x_gt'] = inpaint_300_steps_gt_loss.T
            inpaint_300_steps_avg_loss['mean_gt'] = computeMean(inpaint_300_steps_gt_loss)
            inpaint_300_steps_avg_loss['variance_gt'] = computeVariance(inpaint_300_steps_gt_loss)

            inpaint_300_steps_avg_loss['x_sample'] = inpaint_300_steps_sample_loss.T
            inpaint_300_steps_avg_loss['mean_sample'] = computeMean(inpaint_300_steps_sample_loss)
            inpaint_300_steps_avg_loss['variance_sample'] = computeVariance(inpaint_300_steps_sample_loss)

            inpaint_300_steps_avg_loss['predict_img'] = np.transpose(predict_img_300, (1, 0, 2))
            inpaint_300_steps_avg_loss['gt_img'] = GTImages

            # save all of the model data, so that it can be resumed later.
            if SAVE_DATA:
                np.savez(SAVE_DATA_PATH + '/inpaint_1_' + str(RNN_SIZE) + '_' + str(RNN_LAYERS),
                         x_gt_1=inpaint_1_step_avg_loss['x_gt'],
                         mean_gt_1=inpaint_1_step_avg_loss['mean_gt'],
                         variance_gt_1=inpaint_1_step_avg_loss['variance_gt'],
                         x_sample_1=inpaint_1_step_avg_loss['x_sample'],
                         mean_sample_1=inpaint_1_step_avg_loss['mean_sample'],
                         variance_sample_1=inpaint_1_step_avg_loss['variance_sample'],
                         predict_img_1=inpaint_1_step_avg_loss['predict_img'],
                         gt_img_1=inpaint_1_step_avg_loss['gt_img']
                         )

                np.savez(SAVE_DATA_PATH + '/inpaint_10_' + str(RNN_SIZE) + '_' + str(RNN_LAYERS),
                         x_gt_10=inpaint_10_steps_avg_loss['x_gt'],
                         mean_gt_10=inpaint_10_steps_avg_loss['mean_gt'],
                         variance_gt_10=inpaint_10_steps_avg_loss['variance_gt'],
                         x_sample_10=inpaint_10_steps_avg_loss['x_sample'],
                         mean_sample_10=inpaint_10_steps_avg_loss['mean_sample'],
                         variance_sample_10=inpaint_10_steps_avg_loss['variance_sample'],
                         predict_img_10=inpaint_10_steps_avg_loss['predict_img'],
                         gt_img_10=inpaint_10_steps_avg_loss['gt_img']
                         )

                np.savez(SAVE_DATA_PATH + '/inpaint_28_' + str(RNN_SIZE) + '_' + str(RNN_LAYERS),
                         x_gt_28=inpaint_28_steps_avg_loss['x_gt'],
                         mean_gt_28=inpaint_28_steps_avg_loss['mean_gt'],
                         variance_gt_28=inpaint_28_steps_avg_loss['variance_gt'],
                         x_sample_28=inpaint_28_steps_avg_loss['x_sample'],
                         mean_sample_28=inpaint_28_steps_avg_loss['mean_sample'],
                         variance_sample_28=inpaint_28_steps_avg_loss['variance_sample'],
                         predict_img_28=inpaint_28_steps_avg_loss['predict_img'],
                         gt_img_28=inpaint_28_steps_avg_loss['gt_img']
                         )

                np.savez(SAVE_DATA_PATH + '/inpaint_300_' + str(RNN_SIZE) + '_' + str(RNN_LAYERS),
                         x_gt_300=inpaint_300_steps_avg_loss['x_gt'],
                         mean_gt_300=inpaint_300_steps_avg_loss['mean_gt'],
                         variance_gt_300=inpaint_300_steps_avg_loss['variance_gt'],
                         x_sample_300=inpaint_300_steps_avg_loss['x_sample'],
                         mean_sample_300=inpaint_300_steps_avg_loss['mean_sample'],
                         variance_sample_300=inpaint_300_steps_avg_loss['variance_sample'],
                         predict_img_300=inpaint_300_steps_avg_loss['predict_img'],
                         gt_img_300=inpaint_300_steps_avg_loss['gt_img']
                         )

                df = pd.DataFrame({'xent_gt_1': inpaint_1_step_gt_loss,
                                   'xent_sample_1': inpaint_1_step_sample_loss,
                                   'xent_gt_10': inpaint_10_steps_avg_loss['mean_gt'],
                                   'xent_sample_10': inpaint_10_steps_avg_loss['mean_sample'],
                                   'xent_gt_28': inpaint_28_steps_avg_loss['mean_gt'],
                                   'xent_sample_28': inpaint_28_steps_avg_loss['mean_sample'],
                                   'xent_gt_300': inpaint_300_steps_avg_loss['mean_gt'],
                                   'xent_sample_300': inpaint_300_steps_avg_loss['mean_sample'],
                                   })
                df.to_csv(SAVE_DATA_PATH + "/xentropy_gru_" + str(RNN_SIZE) + '_' + str(RNN_LAYERS) + ".csv",
                          index=False)

            # Visualize
            # visualizeSamples(SAMPLES, inpaint_1_step_avg_loss, inpaint_10_steps_avg_loss, inpaint_28_steps_avg_loss,
            #                 inpaint_300_steps_avg_loss, SAVE_DATA_PATH)
            print('finished processing')


    else:
        # Run mode
        print('Running in run mode..')
        runModel(SAVE_DATA_PATH, SAMPLES, SAVE_DATA_PATH)
        print('-finish-')


# in order to run code from console command: python task2b.py <arg1> <arg2> <arg3> <arg4> <arg5> <arg6>
# arg1 = build / run
# arg2 = lstm / gru
# arg3 = (int) rnn size e.g. 32 or 64 or 128
# arg 4 = (int) number of layers, eg, 1 or 3
# arg 5 = save / nosave
# arg 6 = (comma separate value) sample numbers to run. eg. 1,2,3,4
if __name__ == '__main__':
    args = sys.argv

    if len(args) > 1:
        MODE = args[1].lower() == 'build'
        isLSTM = args[2].lower() == 'lstm'
        RNN_SIZE = args[3]
        RNN_LAYERS = int(args[4])
        SAVE_DATA = args[5].lower() == 'save'
        SAMPLES = str(args[6]).split(',')
    else:
        MODE = False
        isLSTM = False
        RNN_SIZE = '128'
        RNN_LAYERS = 1
        SAVE_DATA = True
        SAMPLES = [9, 62, 94, 90]

    FOLDER = 'task2_'
    if isLSTM and RNN_LAYERS == 1:
        FOLDER += 'lstm_' + RNN_SIZE
    elif isLSTM and RNN_LAYERS > 1:
        FOLDER += 'lstm_' + RNN_SIZE + '_' + str(RNN_LAYERS)
    elif not isLSTM and RNN_LAYERS == 1:
        FOLDER += 'gru_' + RNN_SIZE
    else:
        FOLDER += 'gru_' + RNN_SIZE + '_' + str(RNN_LAYERS)

    main(MODE, isLSTM, int(RNN_SIZE), RNN_LAYERS, FOLDER, SAVE_DATA, SAMPLES)
