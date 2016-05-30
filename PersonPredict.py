"""
Logistic Regression With LBFGS Example.
"""
from __future__ import print_function

import datetime
import gzip
# from importlib import reload

from gensim.models import Word2Vec
# from mpmath.tests.test_elliptic import zero, one

# $example on$
import pandas as pd
import numpy as np
import csv
# $example off$

from Functions import cleanSent, makePredOVO
import csv
import os
import sys
import string
import re
import numpy as np
from gensim.models import Word2Vec

def parsePoint(line, index2word_set, model, num_features):
    ###### IF USING GOOGLE WORD 2 VEC MODEL, PLEASE UNCOMMENT THE FOLLWOING LINE:
    #index2word_set = [name.lower() for name in index2word_set]
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.0
    text = line[0]
    label = 0 #line[1], unlabeled data wont have this column, so putting dummy value
    for word in cleanSent(text[2:]):
        if word and word in index2word_set:
            nwords = nwords + 1.0
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    featureVec = np.nan_to_num(featureVec)
    return LabeledPoint(float(label), featureVec), text

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
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SQLContext # or from pyspark.sql import HiveContext

print('config is done')

sc = SparkContext('local[*]',appName="PythonLogisticRegressionWithLBFGSExample")
sqlContext = SQLContext(sc)
labels = ['Politics','Finance','Sports','Sci&Tech','Entertainment','Crime']
labels_num = [[0.0,1.0],[0.0,2.0],[0.0,3.0],[0.0,4.0],[0.0,5.0],[1.0,2.0],[1.0,3.0],[1.0,4.0],[1.0,5.0],
             [2.0,3.0],[2.0,4.0],[2.0,5.0],[3.0,4.0],[3.0,5.0],[4.0,5.0]]

modelnames = ['pol_fin', 'pol_sports', 'pol_tech', 'pol_ent', 'pol_crime', 'fin_sports', 'fin_tech', 'fin_ent',
              'fin_crime', 'sports_tech', 'sports_ent', 'sports_crime', 'tech_ent', 'tech_crime', 'ent_crime']
path = config.get('Path', 'classificationmodels')
# Build the model
models = []  # load and save models here
for i in range(0, len(modelnames)):
    g = gzip.open(path + modelnames[i] + '.pkl.gz', 'rb')
    print(g)
    print(type(g))
    model = cPickle.load(g)
    print(model)
    models.append(model)
    g.close()

persons = ['DanWetzel', 'DaveDiMartino', 'jaketapper', 'michaelsantoli', 'verge', 'YahooNoise',
           'maddow', 'SuzeOrmanShow', 'justin_fenton' , 'JamieHersch', 'lifehacker', 'TMZ',
           'chinpanchamia', 'SumitGouthaman']
for person in persons:
    data = sc.textFile("persondata/" + person + ".csv")
    data = data.mapPartitions(lambda x: csv.reader(x, delimiter='`', quotechar='|'))
    num_features = 300
    googleModel = False
    if googleModel:
        # model_name = "Models\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin"
        model_name = "Models/GoogleNews-vectors-negative300.bin"
        model = Word2Vec.load_word2vec_format(model_name, binary=True)
        model.init_sims(replace=True)
        index2word_set = set(model.index2word)
        index2word_set = [name.lower() for name in index2word_set]
    else:
        model_name = "Models/ModelforStreaming300_additional_ent"
        model = Word2Vec.load(model_name)
        index2word_set = set(model.index2word)
    print('Model loaded')
    f = lambda j: parsePoint(j, index2word_set,model, num_features)
    parsedData = data.map(f).cache()
    print(parsedData)
    print("NUMBER OF PARTITIONS",parsedData.getNumPartitions())
    print("NUMBER OF ROWS IN RDD",parsedData.count())
    parsedDataDF = parsedData.toDF(['Label&Features','Tweet']).toPandas()

    labelsAndPreds = []
    df = []

    for modellr in models:
        outputrdd = parsedData.map(lambda p: (p[0].label, modellr.predict(p[0].features)))
        labelsAndPreds.append(outputrdd)
        outputdf = outputrdd.toDF(['label', 'prediction']).toPandas()
        df.append(outputdf)

    lab_count = np.zeros((parsedData.count(),len(labels)),dtype="int32")
    for i in range(0,len(models)):
        lab_count = makePredOVO(df[i],labels_num[i],lab_count)

    cz,correct = 0,0
    label_count = np.zeros((len(labels),), dtype="int32")
    for i in range(0,lab_count.shape[0]):
        if np.count_nonzero(lab_count[i,]) > 0:
            args = np.argwhere(lab_count[i,] == np.amax(lab_count[i,]))
            argl = args.flatten().tolist()
            for j in argl:
                label_count[j] += 1
    print(label_count)
    print("The most fitted category for %s is %s" % (person, labels[np.argmax(label_count)]))
    print("===================================")