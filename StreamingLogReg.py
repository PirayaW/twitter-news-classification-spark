"""
Streaming Linear Regression Example.
"""
from __future__ import print_function

# $example on$
import csv
import sys

import datetime
import pandas as pd
import numpy as np
# $example off$
from gensim.models import Word2Vec
from pyspark import SparkContext, SQLContext
from pyspark.mllib.classification import LogisticRegressionWithSGD, StreamingLogisticRegressionWithSGD
from pyspark.streaming import StreamingContext
# $example on$
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD
# $example off$
from Functions import parsePoint, Nlabels2, makePredOVO

if __name__ == "__main__":

    sc = SparkContext('local[*]',appName="PythonLogisticRegressionWithLBFGSExample")
    ssc = StreamingContext(sc, 1)
    sqlContext = SQLContext(sc)

    labels = ['Politics','Finance','Sports','Sci&Tech','Entertainment','Crime']
    labels_num = [[0.0,1.0],[0.0,2.0],[0.0,3.0],[0.0,4.0],[0.0,5.0],[1.0,2.0],[1.0,3.0],[1.0,4.0],[1.0,5.0],
                  [2.0,3.0],[2.0,4.0],[2.0,5.0],[3.0,4.0],[3.0,5.0],[4.0,5.0]]
    data = ssc.textFileStream("/training/data/dir") #SPECIFY THE TRAINING DATA DIRECTORY HERE
    testData = ssc.textFileStream("/testing/data/dir") #SPECIFY THE TESTING DATA DIRECTORY HERE
    data = data.mapPartitions(lambda x: csv.reader(x, delimiter='`', quotechar='|'))
    testData = testData.mapPartitions(lambda x: csv.reader(x, delimiter='`', quotechar='|'))
    #Model details
    num_features = 300
    #model_name = "Models\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin"
    #model = Word2Vec.load_word2vec_format(model_name, binary=True)
    #model.init_sims(replace=True)
    # model_name = "../Models/ModelforStreaming300format_label" # Word2Vec Model
    # model = Word2Vec.load_word2vec_format(model_name)
    model_name = "../Models/ModelforStreaming300_label" # Word2Vec Model
    model = Word2Vec.load(model_name)
    index2word_set = set(model.index2word)
    f = lambda j: parsePoint(j, index2word_set,model, num_features)
    parsedData = data.map(f).cache()
    parsedTestData = testData.map(f).cache()
    print("NUMBER OF PARTITIONS",parsedData.getNumPartitions())
    print("NUMBER OF ROWS IN TRAIN --- TEST RDDs",parsedData.count(),parsedTestData.count())
    parsedDataDF = parsedData.toDF(['Label&Features','Tweet']).toPandas()
    parsedTestDataDF = parsedTestData.toDF(['Label&Features','Tweet']).toPandas()

    # One vs One strategy applied, so 15 models required
    # Problem with creating list of RDD's, see link below: hence, not usings lists
    # http://stackoverflow.com/questions/34788925/unexpected-results-when-making-dicts-and-lists-of-rdds-in-pyspark
    pol_fin = parsedData.filter(lambda p: (p[0].label==0.0 or p[0].label==1.0))
    pol_sports = parsedData.filter(lambda p: (p[0].label==0.0 or p[0].label==2.0))
    pol_tech = parsedData.filter(lambda p: (p[0].label==0.0 or p[0].label==3.0))
    pol_ent = parsedData.filter(lambda p: (p[0].label==0.0 or p[0].label==4.0))
    pol_crime = parsedData.filter(lambda p: (p[0].label==0.0 or p[0].label==5.0))
    fin_sports = parsedData.filter(lambda p: (p[0].label==1.0 or p[0].label==2.0))
    fin_tech = parsedData.filter(lambda p: (p[0].label==1.0 or p[0].label==3.0))
    fin_ent = parsedData.filter(lambda p: (p[0].label==1.0 or p[0].label==4.0))
    fin_crime = parsedData.filter(lambda p: (p[0].label==1.0 or p[0].label==5.0))
    sports_tech = parsedData.filter(lambda p: (p[0].label==2.0 or p[0].label==3.0))
    sports_ent = parsedData.filter(lambda p: (p[0].label==2.0 or p[0].label==4.0))
    sports_crime = parsedData.filter(lambda p: (p[0].label==2.0 or p[0].label==5.0))
    tech_ent = parsedData.filter(lambda p: (p[0].label==3.0 or p[0].label==4.0))
    tech_crime = parsedData.filter(lambda p: (p[0].label==3.0 or p[0].label==5.0))
    ent_crime = parsedData.filter(lambda p: (p[0].label==4.0 or p[0].label==5.0))

    p = lambda j: Nlabels2(j, 0.0)
    f = lambda j: Nlabels2(j, 1.0)
    s = lambda j: Nlabels2(j, 2.0)
    t = lambda j: Nlabels2(j, 3.0)
    e = lambda j: Nlabels2(j, 4.0)
    c = lambda j: Nlabels2(j, 5.0)

    pol_fin = pol_fin.map(p)
    pol_sports = pol_sports.map(p)
    pol_tech = pol_tech.map(p)
    pol_ent = pol_ent.map(p)
    pol_crime = pol_crime.map(p)
    fin_sports = fin_sports.map(f)
    fin_tech = fin_tech.map(f)
    fin_ent = fin_ent.map(f)
    fin_crime = fin_crime.map(f)
    sports_tech = sports_tech.map(s)
    sports_ent = sports_ent.map(s)
    sports_crime = sports_crime.map(s)
    tech_ent = tech_ent.map(t)
    tech_crime = tech_crime.map(t)
    ent_crime = ent_crime.map(e)

    allrdd = [pol_fin,pol_sports,pol_tech,pol_ent,pol_crime,fin_sports,fin_tech,fin_ent,fin_crime,sports_tech,sports_ent,sports_crime,
              tech_ent,tech_crime,ent_crime]
    # Build the model
    # numFeatures = 3
    # model.setInitialWeights([0.0, 0.0, 0.0])
    models = [] #incase needed
    labelsAndPreds = []
    df = []
    for irdd in allrdd:
        print(irdd)
        # modellr = LogisticRegressionWithSGD.train(irdd.map(lambda x: x[0]))
        modellr = StreamingLogisticRegressionWithSGD()
        modellr.trainOn(irdd.map(lambda x: x[0]))
        print(modellr)
        models.append(modellr)
        #outputrdd = parsedData.map(lambda p: (p[0].label, models[i].predict(p[0].features)))
        outputrdd = modellr.predictOnValues(parsedTestData.map(lambda lp: (lp[0].label, lp[0].features)))
        labelsAndPreds.append(outputrdd)
        outputdf = outputrdd.toDF(['label', 'prediction']).toPandas()
        df.append(outputdf)

    lab_count = np.zeros((parsedTestData.count(),len(labels)),dtype="int32")
    for i in range(0,len(allrdd)):
        lab_count = makePredOVO(df[i],labels_num[i],lab_count)

    cz,correct = 0,0
    parsedTestDataDF['PredictedClass'] = pd.np.empty((len(testData), 0)).tolist()
    for i in range(0,lab_count.shape[0]):
        if np.count_nonzero(lab_count[i,])==0:
            cz += 1
            pred_label = "Other"
        else:
            args = np.argwhere(lab_count[i,] == np.amax(lab_count[i,]))
            argl = args.flatten().tolist()
            pred_label = [labels[i] for i in argl]
        parsedTestDataDF.set_value(i, 'PredictedClass', pred_label)
        if parsedTestDataDF.iloc[i]['Label&Features'].label in argl:
            correct += 1
    print(parsedTestDataDF)
    #print("NO CLASS", cz)
    print("ACCURACY",correct/parsedTestDataDF.shape[0])
    cols = ['Tweet','PredictedClass']
    dt = datetime.datetime.strftime(datetime.datetime.now(), '%H%M%S')
    parsedTestDataDF.to_csv('OutputFile'+dt+'.txt',sep=',',columns=cols,index=False)
    print('OutputFile generated')

    ssc.start()
    ssc.awaitTermination()
    # $example off$