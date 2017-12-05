import time
import numpy
import scipy
import cnn_data_engine
try:
    import cPickle as pickle
except:
    import pickle as pickle

def unit_test_svm_problem(libsvmFilename, num_features):
    """
    Unit test on checking the data is correctly input. If the output 
    is the same as the original input, then return true.
    """
    s = time.time()
    y, x = cnn_data_engine.svm_read_problem(libsvmFilename, num_features)
    e = time.time() - s
    print('Time: %g' % e)
    return compare_with_libsvm_data(libsvmFilename, y, x)

def compare_with_libsvm_data(filename, y, x):
    i = 0
    fp = open(filename, 'r')
    lines = fp.readlines()
    for line in lines:
        buf = ''
        buf += str(int(y[i])) + ' '
        for j in range(len(x[i])):
            v = x[i][j]
            if v == 0.0:
                continue
            elif v == 1.0:
                v = 1
            elif v == -1.0:
                v = -1
            buf += str(j + 1) + ':'
            buf += str(v) + ' '
        buf+='\n'
        if (line != buf):
            errmsg = 'line ' + str(i) + ' does not match:\n'
            errmsg += line + '\n'
            errmsg += buf + '\n'
            return errmsg
        i += 1
    return True

def unit_test_output_svm_problem(filename, y, x):
    """
    Output a libsvm format file

    Argument:
    ---------
    y: numpy array
      label vector with one dimension.

    x: numpy array
      A 2d matrix with dimension of number of instance times number of features

    """
    fp = open(filename, 'w')
    for i in range(len(x)):
        fp.write(str(int(y[i])))
        fp.write(' ')
        for j in range(len(x[i])):
            if x[i][j] == 0.0:
                continue
            else:
              fp.write(str(j+1))
              fp.write(':')
              if x[i][j] == 1.0:
                fp.write('1')
              elif x[i][j] == -1.0:
                fp.write('-1')
              else:
                fp.write(str(x[i][j]))
              fp.write(' ')
        fp.write('\n')
    fp.close()

def unit_test_pickle(fileToBeTest, fileToBeCompare):
    """
    Arguments:

    fileToBeTest: str
    fileToBeCompare: str
    """
    import pickle
    filehandler = open(fileToBeTest, 'rb')
    (y, x) = pickle.load(filehandler)
    filehandler.close()
    return compare_with_libsvm_data(fileToBeCompare, y, x)

def unit_test_data_reformat():
    num_inst = 4; channel = 3; height = width = 2;
    x = numpy.array([[1,2,3,4,5,6,7,8,9,10,11,12], [4,3,2,1,4,3,2,1,4,3,2,1], [1,2,3,4,1,2,3,4,1,2,3,4], [1,2,3,4,4,3,2,1,1,2,3,4]])
    output_x = numpy.array([[[[ 1, 5, 9], [ 2, 6,10]],
			     [[ 3, 7,11], [ 4, 8,12]]],
			    [[[ 4, 4, 4], [ 3, 3, 3]],
			     [[ 2, 2, 2], [ 1, 1, 1]]],
			    [[[ 1, 1, 1], [ 2, 2, 2]],
			     [[ 3, 3, 3], [ 4, 4, 4]]],
			    [[[ 1, 4, 1], [ 2, 3, 2]],
			     [[ 3, 2, 3], [ 4, 1, 4]]]])
    test_images = cnn_data_engine.data_reformat(x, channel, height, width)
    return all((test_images == output_x).flatten())

def unit_test_dense_to_one_hot():
    num_classes = 10
    y = numpy.array([3.0, 9.0, 0.0, 1.0])
    output_y = numpy.array([[ 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [ 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [ 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
    labels_one_hot = cnn_data_engine.dense_to_one_hot(y, num_classes)
    return all((labels_one_hot == output_y).flatten())


if __name__=='__main__':
    #################################

    print("Unit test: Read data")
    try:
        libsvmFilename = 'mnist.scale.t'
        num_features = 3*28*28
        #libsvmFile = '/home/loong/data/higgs.scale.t'
        #num_features = 28
        print(unit_test_svm_problem(libsvmFilename, num_features))
    except Exception as e:
        print(e)
    print('\n')

    #################################

    print("Unit test: Load data from pickle")
    try:
        fileToBeTest = 'mnist/test_data.pkl'
        fileToBeCompare = 'mnist.scale.t'
        print(unit_test_pickle(fileToBeTest, fileToBeCompare))
    except Exception as e:
        print(e)
    print('\n')

    #################################

    print("Unit test: Reformat data")
    try:
        print(unit_test_data_reformat())
    except Exception as e:
        print(e)
    print('\n')

    #################################

    print("Unit test: Dense to one hot")
    try:
        print(unit_test_dense_to_one_hot())
    except Exception as e:
        print(e)
    print('\n')

    #################################

    # Input
    print("Unit test: Test dataset.test.next_batch()")
    try:
        TEST_BATCH_SIZE = 100
        CHANNEL = 1
        HEIGHT = 28
        WIDTH = 28
        NUM_CLASSES = 10
        train_path = 'mnist.scale'
        test_path = 'mnist.scale.t'
        dataset = cnn_data_engine.read_dataset(train_path, test_path,
                NUM_CLASSES, CHANNEL, HEIGHT, WIDTH, one_hot=True) # Libsvm format dataset
        test_batch = dataset.test.next_batch(TEST_BATCH_SIZE, withoutMixWithNextEpoch=True, shuffle=False)
        test_batch = dataset.test.next_batch(TEST_BATCH_SIZE, withoutMixWithNextEpoch=True, shuffle=False)
        y = numpy.argmax(test_batch[1], axis=1).flatten()

        unit_test_output_svm_problem('unit_test_svm_prob.txt', y, test_batch[0])
        print('Please test with `vimdiff unit_test_svm_prob.txt %s`' % test_path)
    except Exception as e:
        print(e)
    print('\n')

    #################################
