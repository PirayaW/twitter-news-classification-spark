# News Tweets Classification using Spark Streaming and MLlib
The goal of this project is to classify news tweets into 6 categories: Politics, Finance, Sports, Sci&Tech, Entertainment, and Crime.

CS249: Cloud Computing, Spring 2016
Department of Computer Science
University of California, Los Angeles
## Team
* Piraya Wongnimmarn
* Punit Shetty

## Configuration
* Install Spark on your machine. (http://spark.apache.org/downloads.html)
* Configure you Spark path (SPARK_HOME) in config.properties file. Place this file in the root folder.
```
[Twitter]
consumer_token =
consumer_secret =
access_token =
access_secret =
[Path]
batch_data = ../batch_data/
batch_unlabelled = ../batch_unlabelled/
streaming_data = ../streaming_data/
streaming_test_data = ../streaming_test_data/
streaming_unlabelled_data = ../streaming_unlabelled_data/
streaming_signal = ../streaming_signal/
classificationmodels = ClassificationModels/
json_data = data/
json_persondata = persondata/
persondata = ../persondata/
[Spark]
SPARK_HOME = <your path>/spark-1.6.1-bin-hadoop2.6
python_path = /python
py4j_path = /python/lib/py4j-0.9-src.zip
```
## Online Training
* Run streamingModel.py and streaming/streaming.py concurrently.
* On the streaming module, choose the option to train the model.
* The demo is available at http://www.youtube.com/watch?v=7kG92zHiZKE.

[![Online Training Demo](http://img.youtube.com/vi/7kG92zHiZKE/0.jpg)](http://www.youtube.com/watch?v=7kG92zHiZKE)

## Online Prediction Demo
* Run streamingModel.py and streaming/streaming.py concurrently.
* On the streaming module, choose the option to stream data from batch file or live streaming.
* The demo is available at http://www.youtube.com/watch?v=06SmgyDQfic.
[![Online Prediction Demo](http://img.youtube.com/vi/06SmgyDQfic/0.jpg)](http://www.youtube.com/watch?v=06SmgyDQfic)

## Tools
* Spark 1.6.1 (http://spark.apache.org/docs/latest/index.html)
* Python 2.7
* Tweepy (http://www.tweepy.org)
* Word2Vec (https://code.google.com/archive/p/word2vec/)