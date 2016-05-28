import csv
import os
import sys
import string
import re
import numpy as np
import gzip
from gensim.models import Word2Vec
from bs4 import BeautifulSoup


python_version = sys.version_info.major
if python_version == 3:
    import configparser
    import pickle as cPickle
else:
    import ConfigParser as configparser
    import cPickle


def Nlabels2(line, c):
    if line[0].label == c:
        line[0].label = float(1)
    else:
        line[0].label = float(0.0)
    return line


# Clean Sentence
def cleanSent(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.replace("via", " ")
    text = text.replace("$", "dollar")
    text = BeautifulSoup(text, "lxml").get_text()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split(" ")
    cleaned = []
    for item in tokens:
        if not item.isdigit():  # item not in stop
            item = "".join([e for e in item if e.isalnum()])
            if item:
                cleaned.append(item)
    if cleaned:
        return cleaned
    else:
        return [""]


def parsePoint(line, index2word_set, model, num_features):
    # if googleModel:
    #     ###### IF USING GOOGLE WORD 2 VEC MODEL, PLEASE UNCOMMENT THE FOLLOWING LINE:
    #     index2word_set = [name.lower() for name in index2word_set]
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.0
    text = line[0]
    label = line[1]
    for word in cleanSent(text):
        if word and word in index2word_set:  # (name.upper() for name in USERNAMES)
            # if word and word not in stop and word in index2word_set:
            nwords = nwords + 1.0
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    featureVec = np.nan_to_num(featureVec)
    return LabeledPoint(float(label), featureVec), text

def saveModel(model, name):
    f = gzip.open(name + '.pkl.gz', 'wb')
    cPickle.dump(model, f)
    f.close()


def saveModels(r, model_pol_fin, model_pol_sports, model_pol_tech, model_pol_ent, model_pol_crime,
                    model_fin_sports, model_fin_tech, model_fin_ent, model_fin_crime, model_sports_tech,
                    model_sports_ent, model_sports_crime, model_tech_ent, model_tech_crime, model_ent_crime):
    saveModel(model_pol_fin, "ClassificationModels/pol_fin")
    saveModel(model_pol_sports, "ClassificationModels/pol_sports")
    saveModel(model_pol_tech, "ClassificationModels/pol_tech")
    saveModel(model_pol_ent, "ClassificationModels/pol_ent")
    saveModel(model_pol_crime, "ClassificationModels/pol_crime")
    saveModel(model_fin_sports, "ClassificationModels/fin_sports")
    saveModel(model_fin_tech, "ClassificationModels/fin_tech")
    saveModel(model_fin_ent, "ClassificationModels/fin_ent")
    saveModel(model_fin_crime, "ClassificationModels/fin_crime")
    saveModel(model_sports_tech, "ClassificationModels/sports_tech")
    saveModel(model_sports_ent, "ClassificationModels/sports_ent")
    saveModel(model_sports_crime, "ClassificationModels/sports_crime")
    saveModel(model_tech_ent, "ClassificationModels/tech_ent")
    saveModel(model_tech_crime, "ClassificationModels/tech_crime")
    saveModel(model_ent_crime, "ClassificationModels/ent_crime")
    return 'saved', 1


def predictFunction(r, model_pol_fin, model_pol_sports, model_pol_tech, model_pol_ent, model_pol_crime,
                    model_fin_sports, model_fin_tech, model_fin_ent, model_fin_crime, model_sports_tech,
                    model_sports_ent, model_sports_crime, model_tech_ent, model_tech_crime, model_ent_crime,
                    labels):
    lab_count = np.zeros(6, dtype="int32")
    if model_pol_fin.predict(r[0].features) == 1:
        lab_count[0] += 1
    else:
        lab_count[1] += 1
    if model_pol_sports.predict(r[0].features) == 1:
        lab_count[0] += 1
    else:
        lab_count[2] += 1
    if model_pol_tech.predict(r[0].features) == 1:
        lab_count[0] += 1
    else:
        lab_count[3] += 1
    if model_pol_ent.predict(r[0].features) == 1:
        lab_count[0] += 1
    else:
        lab_count[4] += 1
    if model_pol_crime.predict(r[0].features) == 1:
        lab_count[0] += 1
    else:
        lab_count[5] += 1
    if model_fin_sports.predict(r[0].features) == 1:
        lab_count[1] += 1
    else:
        lab_count[2] += 1
    if model_fin_tech.predict(r[0].features) == 1:
        lab_count[1] += 1
    else:
        lab_count[3] += 1
    if model_fin_ent.predict(r[0].features) == 1:
        lab_count[1] += 1
    else:
        lab_count[4] += 1
    if model_fin_crime.predict(r[0].features) == 1:
        lab_count[1] += 1
    else:
        lab_count[5] += 1
    if model_sports_tech.predict(r[0].features) == 1:
        lab_count[2] += 1
    else:
        lab_count[3] += 1
    if model_sports_ent.predict(r[0].features) == 1:
        lab_count[2] += 1
    else:
        lab_count[4] += 1
    if model_sports_crime.predict(r[0].features) == 1:
        lab_count[2] += 1
    else:
        lab_count[5] += 1
    if model_tech_ent.predict(r[0].features) == 1:
        lab_count[3] += 1
    else:
        lab_count[4] += 1
    if model_tech_crime.predict(r[0].features) == 1:
        lab_count[3] += 1
    else:
        lab_count[5] += 1
    if model_ent_crime.predict(r[0].features) == 1:
        lab_count[4] += 1
    else:
        lab_count[5] += 1
    # print(lab_count)
    correct = 0
    actual_label = labels[int(r[0].label)]
    if int(np.sum(lab_count)) == 0:
        return correct, "Other", actual_label, r[1]
    else:
        args = np.argwhere(lab_count == np.amax(lab_count))
        argl = args.flatten().tolist()
        pred_label = [labels[i] for i in argl]
        if r[0].label in argl:
            correct = 1
        return correct, pred_label, actual_label, r[1]


def calcAccuracy(rdd):
    nrow = rdd.count()
    print(nrow)
    if nrow > 0:
        print("ACCURACY", rdd.filter(lambda x: x[0] == 1).count()/float(nrow))


def printCount(rdd):
    print('Training data size: ', rdd.count())


if __name__ == '__main__':
    config = configparser.RawConfigParser()
    config.read('config.properties')

    # Path for spark source folder
    SPARK_HOME = config.get('Spark', 'SPARK_HOME')
    os.environ['SPARK_HOME'] = SPARK_HOME

    # Append pyspark to Python Path
    sys.path.append(SPARK_HOME + config.get('Spark', 'python_path'))
    sys.path.append(SPARK_HOME + config.get('Spark', 'py4j_path'))

    print('config is done')

    from pyspark import SparkContext, SQLContext
    from pyspark.mllib.classification import StreamingLogisticRegressionWithSGD
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.streaming import StreamingContext

    sc = SparkContext('local[*]', appName="PythonLogisticRegressionWithLBFGSExample")
    ssc = StreamingContext(sc, 30)
    sqlContext = SQLContext(sc)

    labels = ['Politics', 'Finance', 'Sports', 'Sci&Tech', 'Entertainment', 'Crime']
    labels_num = [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0], [0.0, 5.0],
                  [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0],
                  [2.0, 3.0], [2.0, 4.0], [2.0, 5.0],
                  [3.0, 4.0], [3.0, 5.0],
                  [4.0, 5.0]]
    data = ssc.textFileStream("streaming_data/")  # SPECIFY THE TRAINING DATA DIRECTORY HERE
    testData = ssc.textFileStream("streaming_test_data/")  # SPECIFY THE TESTING DATA DIRECTORY HERE
    data = data.mapPartitions(lambda x: csv.reader(x, delimiter='`', quotechar='|'))
    testData = testData.mapPartitions(lambda x: csv.reader(x, delimiter='`', quotechar='|'))

    clear = True
    num_features = 300
    googleModel = False      # change model here
    if googleModel:
        # model_name = "Models\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin"
        model_name = "Models/GoogleNews-vectors-negative300.bin"
        model = Word2Vec.load_word2vec_format(model_name, binary=True)
        # model.init_sims(replace=True)
        index2word_set = set(model.index2word)
        index2word_set = [name.lower() for name in index2word_set]
    else:
        model_name = "Models/ModelforStreaming300_additional"  # Word2Vec Model
        model = Word2Vec.load(model_name)
        index2word_set = set(model.index2word)
    print('Word2Vec Model is loaded')
    # print(model.most_similar(positive=['woman', 'king'], negative=['man']))
    f = lambda j: parsePoint(j, index2word_set, model, num_features)
    parsedData = data.map(f).cache()
    parsedTestData = testData.map(f).cache()

    parsedData.pprint()
    parsedData.foreachRDD(printCount)

    pol_fin = parsedData.filter(lambda p: (p[0].label == 0.0 or p[0].label == 1.0))
    pol_sports = parsedData.filter(lambda p: (p[0].label == 0.0 or p[0].label == 2.0))
    pol_tech = parsedData.filter(lambda p: (p[0].label == 0.0 or p[0].label == 3.0))
    pol_ent = parsedData.filter(lambda p: (p[0].label == 0.0 or p[0].label == 4.0))
    pol_crime = parsedData.filter(lambda p: (p[0].label == 0.0 or p[0].label == 5.0))
    fin_sports = parsedData.filter(lambda p: (p[0].label == 1.0 or p[0].label == 2.0))
    fin_tech = parsedData.filter(lambda p: (p[0].label == 1.0 or p[0].label == 3.0))
    fin_ent = parsedData.filter(lambda p: (p[0].label == 1.0 or p[0].label == 4.0))
    fin_crime = parsedData.filter(lambda p: (p[0].label == 1.0 or p[0].label == 5.0))
    sports_tech = parsedData.filter(lambda p: (p[0].label == 2.0 or p[0].label == 3.0))
    sports_ent = parsedData.filter(lambda p: (p[0].label == 2.0 or p[0].label == 4.0))
    sports_crime = parsedData.filter(lambda p: (p[0].label == 2.0 or p[0].label == 5.0))
    tech_ent = parsedData.filter(lambda p: (p[0].label == 3.0 or p[0].label == 4.0))
    tech_crime = parsedData.filter(lambda p: (p[0].label == 3.0 or p[0].label == 5.0))
    ent_crime = parsedData.filter(lambda p: (p[0].label == 4.0 or p[0].label == 5.0))

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

    model_pol_fin = StreamingLogisticRegressionWithSGD()
    model_pol_sports = StreamingLogisticRegressionWithSGD()
    model_pol_tech = StreamingLogisticRegressionWithSGD()
    model_pol_ent = StreamingLogisticRegressionWithSGD()
    model_pol_crime = StreamingLogisticRegressionWithSGD()
    model_fin_sports = StreamingLogisticRegressionWithSGD()
    model_fin_tech = StreamingLogisticRegressionWithSGD()
    model_fin_ent = StreamingLogisticRegressionWithSGD()
    model_fin_crime = StreamingLogisticRegressionWithSGD()
    model_sports_tech = StreamingLogisticRegressionWithSGD()
    model_sports_ent = StreamingLogisticRegressionWithSGD()
    model_sports_crime = StreamingLogisticRegressionWithSGD()
    model_tech_ent = StreamingLogisticRegressionWithSGD()
    model_tech_crime = StreamingLogisticRegressionWithSGD()
    model_ent_crime = StreamingLogisticRegressionWithSGD()

    if clear:
        model_pol_fin.setInitialWeights([0.0] * num_features)
        model_pol_sports.setInitialWeights([0.0] * num_features)
        model_pol_tech.setInitialWeights([0.0] * num_features)
        model_pol_ent.setInitialWeights([0.0] * num_features)
        model_pol_crime.setInitialWeights([0.0] * num_features)
        model_fin_sports.setInitialWeights([0.0] * num_features)
        model_fin_tech.setInitialWeights([0.0] * num_features)
        model_fin_ent.setInitialWeights([0.0] * num_features)
        model_fin_crime.setInitialWeights([0.0] * num_features)
        model_sports_tech.setInitialWeights([0.0] * num_features)
        model_sports_ent.setInitialWeights([0.0] * num_features)
        model_sports_crime.setInitialWeights([0.0] * num_features)
        model_tech_ent.setInitialWeights([0.0] * num_features)
        model_tech_crime.setInitialWeights([0.0] * num_features)
        model_ent_crime.setInitialWeights([0.0] * num_features)
    else:
        model_path = config.get('Path', 'classificationmodels')
        with gzip.open(model_path + 'pol_fin.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_pol_fin.setInitialWeights(model.weights)
        with gzip.open(model_path + 'pol_sports.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_pol_sports.setInitialWeights(model.weights)
        with gzip.open(model_path + 'pol_tech.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_pol_tech.setInitialWeights(model.weights)
        with gzip.open(model_path + 'pol_ent.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_pol_ent.setInitialWeights(model.weights)
        with gzip.open(model_path + 'pol_crime.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_pol_crime.setInitialWeights(model.weights)
        with gzip.open(model_path + 'fin_sports.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_fin_sports.setInitialWeights(model.weights)
        with gzip.open(model_path + 'fin_tech.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_fin_tech.setInitialWeights(model.weights)
        with gzip.open(model_path + 'fin_ent.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_fin_ent.setInitialWeights(model.weights)
        with gzip.open(model_path + 'fin_crime.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_fin_crime.setInitialWeights(model.weights)
        with gzip.open(model_path + 'sports_tech.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_sports_tech.setInitialWeights(model.weights)
        with gzip.open(model_path + 'sports_ent.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_sports_ent.setInitialWeights(model.weights)
        with gzip.open(model_path + 'sports_crime.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_sports_crime.setInitialWeights(model.weights)
        with gzip.open(model_path + 'tech_ent.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_tech_ent.setInitialWeights(model.weights)
        with gzip.open(model_path + 'tech_crime.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_tech_crime.setInitialWeights(model.weights)
        with gzip.open(model_path + 'ent_crime.pkl.gz', 'rb') as g:
            model = cPickle.load(g)
            model_ent_crime.setInitialWeights(model.weights)
    print(model_ent_crime.latestModel().weights)

    model_pol_fin.trainOn(pol_fin.map(lambda x: x[0]))
    model_pol_sports.trainOn(pol_sports.map(lambda x: x[0]))
    model_pol_tech.trainOn(pol_tech.map(lambda x: x[0]))
    model_pol_ent.trainOn(pol_ent.map(lambda x: x[0]))
    model_pol_crime.trainOn(pol_crime.map(lambda x: x[0]))
    model_fin_sports.trainOn(fin_sports.map(lambda x: x[0]))
    model_fin_tech.trainOn(fin_tech.map(lambda x: x[0]))
    model_fin_ent.trainOn(fin_ent.map(lambda x: x[0]))
    model_fin_crime.trainOn(fin_crime.map(lambda x: x[0]))
    model_sports_tech.trainOn(sports_tech.map(lambda x: x[0]))
    model_sports_ent.trainOn(sports_ent.map(lambda x: x[0]))
    model_sports_crime.trainOn(sports_crime.map(lambda x: x[0]))
    model_tech_ent.trainOn(tech_ent.map(lambda x: x[0]))
    model_tech_crime.trainOn(tech_crime.map(lambda x: x[0]))
    model_ent_crime.trainOn(ent_crime.map(lambda x: x[0]))

    output = parsedTestData.map(lambda r: predictFunction(r,
                                                 model_pol_fin.latestModel(),
                                                 model_pol_sports.latestModel(),
                                                 model_pol_tech.latestModel(),
                                                 model_pol_ent.latestModel(),
                                                 model_pol_crime.latestModel(),
                                                 model_fin_sports.latestModel(),
                                                 model_fin_tech.latestModel(),
                                                 model_fin_ent.latestModel(),
                                                 model_fin_crime.latestModel(),
                                                 model_sports_tech.latestModel(),
                                                 model_sports_ent.latestModel(),
                                                 model_sports_crime.latestModel(),
                                                 model_tech_ent.latestModel(),
                                                 model_tech_crime.latestModel(),
                                                 model_ent_crime.latestModel(),
                                                 labels
                                                 ))

    output.pprint()
    output.foreachRDD(calcAccuracy)
    output.saveAsTextFiles("Output/testData")

    count = parsedData.map(lambda r: ('merged', 1)).reduceByKey(lambda a, b: a+b)
    count = count.map(lambda r: saveModels(r,
                                        model_pol_fin.latestModel(),
                                        model_pol_sports.latestModel(),
                                        model_pol_tech.latestModel(),
                                        model_pol_ent.latestModel(),
                                        model_pol_crime.latestModel(),
                                        model_fin_sports.latestModel(),
                                        model_fin_tech.latestModel(),
                                        model_fin_ent.latestModel(),
                                        model_fin_crime.latestModel(),
                                        model_sports_tech.latestModel(),
                                        model_sports_ent.latestModel(),
                                        model_sports_crime.latestModel(),
                                        model_tech_ent.latestModel(),
                                        model_tech_crime.latestModel(),
                                        model_ent_crime.latestModel()
                                        )).reduceByKey(lambda a, b: a+b)
    count.pprint()  # so that the map function get executed

    ssc.start()
    ssc.awaitTermination()
