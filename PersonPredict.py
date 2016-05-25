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

from Functions import cleanSent, parsePoint, makePredFile, Nlabels2, makePredOVO, ClassPostivePredictions
import csv
import os
import sys
import string
import re
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
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SQLContext # or from pyspark.sql import HiveContext

# # Path for spark source folder
# SPARK_HOME = config.get('Spark', 'SPARK_HOME')
# print(SPARK_HOME)
# os.environ['SPARK_HOME'] = SPARK_HOME
# # Append pyspark to Python Path
# print(config.get('Spark', 'python_path'))
# print(config.get('Spark', 'py4j_path'))
# sys.path.append(config.get('Spark', 'python_path'))
# sys.path.append(config.get('Spark', 'py4j_path'))
print('config is done')

sc = SparkContext('local[*]',appName="PythonLogisticRegressionWithLBFGSExample")
sqlContext = SQLContext(sc)
labels = ['Politics','Finance','Sports','Sci&Tech','Entertainment','Crime']
labels_num = [[0.0,1.0],[0.0,2.0],[0.0,3.0],[0.0,4.0],[0.0,5.0],[1.0,2.0],[1.0,3.0],[1.0,4.0],[1.0,5.0],
             [2.0,3.0],[2.0,4.0],[2.0,5.0],[3.0,4.0],[3.0,5.0],[4.0,5.0]]
# data = sc.textFile("persondata/YahooNoise.csv")
data = sc.textFile("batch_data/20160501_00.csv")
data = data.mapPartitions(lambda x: csv.reader(x, delimiter='`', quotechar='|'))
num_features = 300
#model_name = "E:\\Punit\\D\\UCLA\\Spring16\\MSProject\\Models\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin"
#model = Word2Vec.load_word2vec_format(model_name, binary=True)
#model.init_sims(replace=True)
model_name = "Models/ModelforStreaming300_label"
model = Word2Vec.load(model_name)
index2word_set = set(model.index2word)
f = lambda j: parsePoint(j, index2word_set,model, num_features)
parsedData = data.map(f).cache()
print(parsedData)
print("NUMBER OF PARTITIONS",parsedData.getNumPartitions())
print("NUMBER OF ROWS IN RDD",parsedData.count())
parsedDataDF = parsedData.toDF(['Label&Features','Tweet']).toPandas()

# pol_fin = parsedData.filter(lambda p: (p[0].label==0.0 or p[0].label==1.0))
# pol_sports = parsedData.filter(lambda p: (p[0].label==0.0 or p[0].label==2.0))
# pol_tech = parsedData.filter(lambda p: (p[0].label==0.0 or p[0].label==3.0))
# pol_ent = parsedData.filter(lambda p: (p[0].label==0.0 or p[0].label==4.0))
# pol_crime = parsedData.filter(lambda p: (p[0].label==0.0 or p[0].label==5.0))
# fin_sports = parsedData.filter(lambda p: (p[0].label==1.0 or p[0].label==2.0))
# fin_tech = parsedData.filter(lambda p: (p[0].label==1.0 or p[0].label==3.0))
# fin_ent = parsedData.filter(lambda p: (p[0].label==1.0 or p[0].label==4.0))
# fin_crime = parsedData.filter(lambda p: (p[0].label==1.0 or p[0].label==5.0))
# sports_tech = parsedData.filter(lambda p: (p[0].label==2.0 or p[0].label==3.0))
# sports_ent = parsedData.filter(lambda p: (p[0].label==2.0 or p[0].label==4.0))
# sports_crime = parsedData.filter(lambda p: (p[0].label==2.0 or p[0].label==5.0))
# tech_ent = parsedData.filter(lambda p: (p[0].label==3.0 or p[0].label==4.0))
# tech_crime = parsedData.filter(lambda p: (p[0].label==3.0 or p[0].label==5.0))
# ent_crime = parsedData.filter(lambda p: (p[0].label==4.0 or p[0].label==5.0))

# p = lambda j: Nlabels2(j, 0.0)
# f = lambda j: Nlabels2(j, 1.0)
# s = lambda j: Nlabels2(j, 2.0)
# t = lambda j: Nlabels2(j, 3.0)
# e = lambda j: Nlabels2(j, 4.0)
# c = lambda j: Nlabels2(j, 5.0)
#
# pol_fin = pol_fin.map(p)
# pol_sports = pol_sports.map(p)
# pol_tech = pol_tech.map(p)
# pol_ent = pol_ent.map(p)
# pol_crime = pol_crime.map(p)
# fin_sports = fin_sports.map(f)
# fin_tech = fin_tech.map(f)
# fin_ent = fin_ent.map(f)
# fin_crime = fin_crime.map(f)
# sports_tech = sports_tech.map(s)
# sports_ent = sports_ent.map(s)
# sports_crime = sports_crime.map(s)
# tech_ent = tech_ent.map(t)
# tech_crime = tech_crime.map(t)
# ent_crime = ent_crime.map(e)
# allrdd = [pol_fin,pol_sports,pol_tech,pol_ent,pol_crime,fin_sports,fin_tech,fin_ent,fin_crime,sports_tech,sports_ent,sports_crime,
#             tech_ent,tech_crime,ent_crime]

modelnames= ['pol_fin','pol_sports','pol_tech','pol_ent','pol_crime','fin_sports','fin_tech','fin_ent','fin_crime',
             'sports_tech','sports_ent','sports_crime','tech_ent','tech_crime','ent_crime']
path = config.get('Path', 'classificationmodels')
# Build the model
models = [] #load and save models here
labelsAndPreds = []
df = []
for i in range(0,len(modelnames)):
    g = gzip.open(path+modelnames[i]+'.pkl.gz', 'rb')
    print(g)
    print(type(g))
    model = cPickle.load(g)
    print(model)
    models.append(model)
    g.close()

for modellr in models:
    #modellr = LogisticRegressionWithSGD.train(irdd.map(lambda x: x[0]))
    #models.append(modellr)
    print(modellr)
    outputrdd = parsedData.map(lambda p: (p[0].label, modellr.predict(p[0].features)))
    labelsAndPreds.append(outputrdd)
    outputdf = outputrdd.toDF(['label', 'prediction']).toPandas()
    df.append(outputdf)

lab_count = np.zeros((parsedData.count(),len(labels)),dtype="int32")
for i in range(0,len(models)):
    lab_count = makePredOVO(df[i],labels_num[i],lab_count)

cz,correct = 0,0
label_count = np.zeros((len(labels),),dtype="int32")
for i in range(0,lab_count.shape[0]):
    if np.count_nonzero(lab_count[i,]) > 0:
        args = np.argwhere(lab_count[i,] == np.amax(lab_count[i,]))
        argl = args.flatten().tolist()
        for i in argl:
            label_count[i] +=1
print(label_count)
print("The most fitted category for this person is %s" %labels[np.argmax(label_count)])