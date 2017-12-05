import sys
import argparse
import numpy
import time
import os
try:
    import cPickle as pickle
except:
    import pickle
from dnn_data_engine import svm_read_problem

def ArgsParser():
  class MyParser(argparse.ArgumentParser):
    def error(self, message):
      sys.stderr.write('error: %s\n' % message)
      self.print_help()
      sys.exit(2)

  parser = MyParser()
  parser.add_argument('-f', '--num_features', help='Number of features', required=True, type=int)
  parser.add_argument('-train', '--train_path', help='specify train path', default='', type=str)
  parser.add_argument('-test', '--test_path', help='specify test path', default='', type=str)
  parser.add_argument('-d', '--destination', help='specify the path you would like to save the output file', default='../data', type=str)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
    args = ArgsParser()
    train_path = args.train_path
    test_path = args.test_path
    num_features = args.num_features
    destination = args.destination

    if (~os.path.isdir(destination)):
        os.system("mkdir -p " + destination)

    print('Convert train data')
    try:
        # Read libsvm data
        s = time.time()
        y_train, x_train = svm_read_problem(train_path, num_features)
        e = time.time() - s
        print('Time: %g' % e)
        x_train = numpy.array(x_train)
        y_train = numpy.array(y_train)

        # Save data using pickle
        with open(destination + '/train_data.pkl', 'wb') as filehandler:
            pickle.dump((y_train, x_train), filehandler)
    except Exception as e:
        print(str(e))

    print('Convert test data')
    try:
        s = time.time()
        y_test, x_test = svm_read_problem(test_path, num_features)
        e = time.time() - s
        print('Time: %g' % e)
        x_test = numpy.array(x_test)
        y_test = numpy.array(y_test)

        # Save data using pickle
        with open(destination + '/test_data.pkl', 'wb') as filehandler:
            pickle.dump((y_test, x_test), filehandler)
    except Exception as e:
        print(str(e))
