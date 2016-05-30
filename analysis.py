from __future__ import print_function

import gzip
from Functions import parsePoint
import csv
import sys
import numpy as np
from gensim.models import Word2Vec

python_version = sys.version_info.major
if python_version == 3:
    import configparser
    import pickle as cPickle
else:
    import ConfigParser as configparser
    import cPickle

config = configparser.RawConfigParser()
config.read('config.properties')

# pyspark need to be imported after import Functions, which sets path for pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext # or from pyspark.sql import HiveContext

print('config is done')

sc = SparkContext('local[*]', appName="Analysis")
sqlContext = SQLContext(sc)
labels = ['Politics', 'Finance', 'Sports', 'Sci&Tech', 'Entertainment', 'Crime']
labels_num = [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0], [0.0, 5.0],
              [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0],
              [2.0, 3.0], [2.0, 4.0], [2.0, 5.0],
              [3.0, 4.0], [3.0, 5.0], [4.0, 5.0]]
data = sc.textFile("batch_data/20160521_12.csv")
data = data.mapPartitions(lambda x: csv.reader(x, delimiter='`', quotechar='|'))
num_features = 300
model_name = "Models/ModelforStreaming300_additional_ent"
model = Word2Vec.load(model_name)
index2word_set = set(model.index2word)
f = lambda j: parsePoint(j, index2word_set, model, num_features)
parsedData = data.map(f).cache()
print(parsedData)
print("NUMBER OF PARTITIONS", parsedData.getNumPartitions())
print("NUMBER OF ROWS IN RDD", parsedData.count())
parsedDataDF = parsedData.toDF(['Label&Features', 'Tweet']).toPandas()

modelnames= ['pol_fin', 'pol_sports', 'pol_tech', 'pol_ent', 'pol_crime', 'fin_sports', 'fin_tech', 'fin_ent',
             'fin_crime', 'sports_tech', 'sports_ent', 'sports_crime', 'tech_ent', 'tech_crime', 'ent_crime']
path = config.get('Path', 'classificationmodels')
# Build the model
models = [] #load and save models here
labelsAndPreds = []
df = []
for i in range(0, len(modelnames)):
    g = gzip.open(path+modelnames[i]+'.pkl.gz', 'rb')
    print(g)
    print(type(g))
    model = cPickle.load(g)
    print(model)
    models.append(model)
    g.close()

for modellr in models:
    print(modellr)
    outputrdd = parsedData.map(lambda p: (p[0].label, modellr.predict(p[0].features)))
    labelsAndPreds.append(outputrdd)
    outputdf = outputrdd.toDF(['label', 'prediction']).toPandas()
    df.append(outputdf)


def makePredOVO_(df, num, lab_count):
    idx = df[df.prediction == 1].index.tolist()
    lab_count[idx, num[0]] += 1
    idx = df[df.prediction == 0].index.tolist()
    lab_count[idx, num[1]] += 1
    return df.label, lab_count


lab_count = np.zeros((parsedData.count(), len(labels)), dtype="int32")
for i in range(0,len(models)):
    labels_, lab_count = makePredOVO_(df[i], labels_num[i], lab_count)
print(labels_)
print(lab_count)

print('Confusion Matrix')
confusionMatrix = np.zeros((len(labels), len(labels)+1), dtype="int32")
correct = 0
for i in range(len(lab_count)):
    actual_label = int(labels_[i])
    lc = lab_count[i]
    if int(np.sum(lc)) == 0:
        confusionMatrix[actual_label][len(labels)] += 1
    else:
        args = np.argwhere(lc == np.amax(lc))
        argl = args.flatten().tolist()
        pred_label = [labels[i] for i in argl]
        # print(pred_label, labels[actual_label])
        if actual_label in argl:
            correct += 1
        for j in argl:
            confusionMatrix[actual_label][j] += 1

print('--------- CONFUSION MATRIX ------------')
print(labels + ['Other'])
print(confusionMatrix)
print('ACCURACY', float(correct)/len(lab_count))
print('Done')
