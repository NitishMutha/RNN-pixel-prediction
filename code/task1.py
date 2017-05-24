import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt


# main code
def main(TRAIN_MODE=True, isLSTM=True, RNN_SIZE=32, RNN_LAYERS=1, FOLDER='task1_a_32'):
    data_dir = '../data/'
    from tensorflow.examples.tutorials.mnist import input_data

    # import dataset with one-hot class encoding
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    def plotGraph(data, label, xlabel, ylabel, title, filename, no):
        plt.figure(no)
        plt.plot(data, label=label)
        plt.grid()
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.title(title)
        plt.legend()
        plt.savefig(filename, format='eps', dpi=1000)
        # plt.show()

    def binarize(images, threshold=0.1):
        return (threshold < images).astype('float')

    def processData(images):
        for im in range(len(images)):
            images[im] = binarize(images[im])
        return images

    processData(mnist.train.images)
    processData(mnist.test.images)

    # network parameters

    IMAGE_DIM = 1
    TIME_STEP = 784
    OUTPUT_NODES = 10
    HIDDEN_LAYER = 100
    # RNN_LAYERS = 1

    # isLSTM = True  # False for GRU

    EPOCH = 50
    BATCH = 256
    LEARNING_RATE = 0.001  # 0.0003
    DISPLAY_STEP = 100
    KEEP_PROB = .50
    EPSILON = 1e-3
    CLIP = 5.

    # RNN_SIZE = 32
    # TRAIN_MODE = True

    accuracyList = []
    errorList = []

    testAcc = []
    testLosses = []

    if isLSTM:
        modelType = 'lstm'
    else:
        modelType = 'gru'

    # FOLDER = 'task1_a_32'

    LOG_PATH = '../summary/' + FOLDER + '/summary'
    MODEL_PATH = '../models/' + FOLDER

    writer = tf.summary.FileWriter(LOG_PATH)

    # Build compuatation graph

    x = tf.placeholder(tf.float32, [None, TIME_STEP, IMAGE_DIM])
    y = tf.placeholder(tf.float32, [None, OUTPUT_NODES])

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

            output = tf.transpose(output, [1, 0, 2])
            last = tf.gather(output, int(output.get_shape()[0]) - 1)

            weight100 = tf.Variable(tf.truncated_normal([RNN_SIZE, HIDDEN_LAYER]))
            # bias100 = tf.Variable(tf.constant(0.1, shape=[1, HIDDEN_LAYER]))
            z = tf.matmul(last, weight100)

            # batch normalization
            batch_mean, batch_var = tf.nn.moments(z, [0])
            scale = tf.Variable(tf.ones([HIDDEN_LAYER]))
            beta = tf.Variable(tf.zeros([HIDDEN_LAYER]))
            batchNorm = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, EPSILON)

            # ReLU
            r1 = tf.nn.relu(batchNorm)
            d1 = tf.nn.dropout(r1, KEEP_PROB)

            weight10 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER, OUTPUT_NODES]))
            bias10 = tf.Variable(tf.constant(0.1, shape=[1, OUTPUT_NODES]))
            z2 = tf.matmul(r1, weight10) + bias10

            # prediction = tf.nn.softmax(tf.matmul(r1, weight10) + bias10)

            return z2

    prediction = RNN(x, isLSTM, RNN_LAYERS, "rnn")

    with tf.name_scope("Loss"):
        # Softmax loss and L2
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
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
    merged_summary = tf.summary.merge_all()

    # Run the graph

    with tf.Session() as sess:
        sess.run(init)
        writer.add_graph(sess.graph)

        text_file = open("../Plots/Output" + FOLDER + ".txt", "w")

        if TRAIN_MODE:
            print('\n\n---Running in train mode---')
            for i in range(EPOCH * BATCH):

                batch_xs, batch_ys = mnist.train.next_batch(BATCH)
                batch_xs = batch_xs.reshape((BATCH, TIME_STEP, IMAGE_DIM))

                _, summary = sess.run([optimizer, merged_summary], feed_dict={x: batch_xs, y: batch_ys})
                writer.add_summary(summary, i)
                if i % BATCH == 0:

                    # saver.save(sess, 'saver/task1_32/lstm', global_step=i)

                    train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})
                    accuracyList.append(train_accuracy)
                    print("Epoch: %d, Training Accuracy: %g" % (int(i / BATCH), train_accuracy))

                    loss = cost.eval(feed_dict={x: batch_xs, y: batch_ys})
                    errorList.append(loss)
                    print("Training Loss: %g \n" % (loss))

                    text_file.write("Epoch: "+ str(int(i / BATCH)) +" Training Accuracy: " + str(train_accuracy) + " | Training Loss: " + str(loss) +" \n");

                    if train_accuracy > .59:
                        saveAcc = tf.Variable(accuracyList, name="accuracyList")
                        errLoss = tf.Variable(errorList, name="errorList")
                        saver.save(sess, MODEL_PATH + '/' + str(int(i / BATCH)) + '_' + modelType + '.ckpt')

            print("\nTraining Complete. Running tests...")

            total_test_batches = int(mnist.test.num_examples / BATCH)
            for j in range(total_test_batches):
                batch_testImage, batch_testLabel = mnist.test.next_batch(BATCH)
                batch_testImage = batch_testImage.reshape((-1, TIME_STEP, IMAGE_DIM))

                y_true = np.argmax(batch_testLabel, 1)
                y_p = tf.argmax(prediction, 1)

                acc, pred = sess.run([accuracy, y_p], feed_dict={x: batch_testImage, y: batch_testLabel})
                print("Batch Test Accuracy: %g" % acc)
                testAcc.append(acc)
                testLoss = cost.eval(feed_dict={x: batch_testImage, y: batch_testLabel})
                print("Batch Test Loss: %g" % testLoss)
                testLosses.append(testLoss)

        else:
            # test mode
            print('\n\n---Running in test mode---')
            saver.restore(sess, MODEL_PATH + '/model.ckpt')

            total_test_batches = int(mnist.test.num_examples / BATCH)
            for j in range(total_test_batches):
                batch_testImage, batch_testLabel = mnist.test.next_batch(BATCH)
                batch_testImage = batch_testImage.reshape((-1, TIME_STEP, IMAGE_DIM))

                y_true = np.argmax(batch_testLabel, 1)
                y_p = tf.argmax(prediction, 1)

                acc, pred = sess.run([accuracy, y_p], feed_dict={x: batch_testImage, y: batch_testLabel})
                print("Batch Test Accuracy: %g" % acc)
                testAcc.append(acc)
                testLoss = cost.eval(feed_dict={x: batch_testImage, y: batch_testLabel})
                print("Batch Test Loss: %g" % testLoss)
                testLosses.append(testLoss)

        if TRAIN_MODE:
            save_path = saver.save(sess, MODEL_PATH + '/model.ckpt')
            print('Model weights saved in file: ', save_path)

        print('Test Accuracy: %g' % np.mean(testAcc))
        print('Test Losses: %g' % np.mean(testLosses))


        text_file.write('Test Accuracy: ' +  str(np.mean(testAcc)) + ' | ' + 'Test Losses: ' +  str(np.mean(testLosses)))
        text_file.close()


# in order to run code from console command: python task1.py <arg1> <arg2> <arg3> <arg4>
# arg1 = train, test
# arg2 = lstm, gru
# arg3 = (int) rnn size e.g. 32 or 64 or 128
# arg 4 = (int) number of layers, eg, 1 or 3
if __name__ == '__main__':
    args = sys.argv

    if len(args) > 1:
        TRAIN_MODE = args[1].lower() == 'train'
        isLSTM = args[2].lower() == 'lstm'
        RNN_SIZE = args[3]
        RNN_LAYERS = int(args[4])
    else:
        TRAIN_MODE = True
        isLSTM = True
        RNN_SIZE = '32'
        RNN_LAYERS = 1

    FOLDER = 'task1_'
    if isLSTM and RNN_LAYERS == 1:
        FOLDER += 'a_' + RNN_SIZE
    elif isLSTM and RNN_LAYERS > 1:
        FOLDER += 'c_' + RNN_SIZE
    elif not isLSTM and RNN_LAYERS == 1:
        FOLDER += 'b_' + RNN_SIZE
    else:
        FOLDER += 'd_' + RNN_SIZE

    main(TRAIN_MODE, isLSTM, int(RNN_SIZE), RNN_LAYERS, FOLDER)
