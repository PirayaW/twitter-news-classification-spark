import os
import sys
import cPickle
import gzip
import random


python_version = sys.version_info.major
if python_version == 3:
    import configparser
else:
    import ConfigParser as configparser


if __name__ == '__main__':
    config = configparser.RawConfigParser()
    config.read('../config.properties')

    # Path for spark source folder
    SPARK_HOME = config.get('Spark', 'SPARK_HOME')
    os.environ['SPARK_HOME'] = SPARK_HOME

    # Append pyspark to Python Path
    sys.path.append(SPARK_HOME + config.get('Spark', 'python_path'))
    sys.path.append(SPARK_HOME + config.get('Spark', 'py4j_path'))

    print('config is done')

    from pyspark import SparkContext

    sc = SparkContext('local[*]', appName="TestModelLoading")
    num_features = 300

    models = [None, None]
    g = gzip.open('../ClassificationModels/pol_fin.pkl.gz', 'rb')
    models[0] = cPickle.load(g)
    g.close()
    g = gzip.open('../ClassificationModels/sports_tech.pkl.gz', 'rb')
    models[1] = cPickle.load(g)
    g.close()

    for i in range(len(models)):
        print(models[i])

    for i in range(10):
        data = [random.random()]*num_features
        for j in range(len(models)):
            predicted = models[j].predict(data)
            print(predicted)

