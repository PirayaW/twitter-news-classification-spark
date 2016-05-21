import string

import nltk
import numpy as np
import re
from bs4 import BeautifulSoup
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint

def Nlabels2(line,c):
        if line[0].label == c:
            line[0].label = float(1)
        else:
            line[0].label = float(0.0)
        return line

# Clean Sentence
def cleanSent(text):
    #text = " ".join(text)
    text = text.lower()
    # remove RT @username
    # text = re.sub(r'RT @\w+\s?:', '', text)
    # remove url
    text = re.sub(r'http\S+', '', text)
    # remove user mentions
    # text = re.sub(r'@\w+\s?', '', text)
    text = text.replace("via"," ")
    text = text.replace("$","dollar")
    text = BeautifulSoup(text,"lxml").get_text()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split(" ")#nltk.word_tokenize(text)
    cleaned = []
    for item in tokens:
        if not item.isdigit(): #item not in stop
            item = "".join([e for e in item if e.isalnum()])
            if item:
                cleaned.append(item)
    if cleaned:
        return cleaned
    else:
        return [""]

# def clean(text):
# 	# remove RT @username
# 	text = re.sub(r'RT @\w+\s?:', '', text)
# 	# replace '&' symbol
# 	text = re.sub(r'&amp', '', text)
# 	# remove '#' symbol before the hashtag
# 	text = re.sub(r'#', '', text)
# 	# remove url
# 	text = re.sub(r'http\S+', '', text)
# 	# remove user mentions
# 	text = re.sub(r'@\w+\s?', '', text)
# 	# ignore non-ascii character
# 	text = text.encode('ascii', 'ignore')
# 	# remove apostrophe
# 	text = re.sub(r'\'', '', text)
# 	# remove comma between numbers
# 	text = re.sub("(?<=[\\d])(,)(?=[\\d])", "", text)
# 	# replace '$' -> ' dollarsign '
# 	text = re.sub(r'\$',' dollarsign ', text)
# 	return text

def parsePoint(line, index2word_set,model, num_features ):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.0
    text = line[0]
    label = line[1]
    if label == '':
        print(line[0])
    #print(cleanSent(text[3:]))
    for word in cleanSent(text[3:]):
        if word and word in index2word_set: #(name.upper() for name in USERNAMES)
        #if word and word not in stop and word in index2word_set:
            nwords = nwords + 1.0
            featureVec = np.add(featureVec,model[word])
    ##### INCLUDE THIS PART IF LABEL INTO PICTURE
    # if line[3] in index2word_set:
    #     nwords = nwords + 1.0
    #     featureVec = np.add(featureVec,model[word])
    #####################################################
    #print("word count",nwords)
    featureVec = np.divide(featureVec,nwords)
    featureVec = np.nan_to_num(featureVec)
    return LabeledPoint(float(label),featureVec),text

def makePredFile(df):
    idx = df[df.prediction==1.0].index.tolist()
    return idx

def makePredOVO(df,num,lab_count):
    idx = df[df.prediction==1.0].index.tolist()
    lab_count[idx,num[0]] += 1
    idx = df[df.prediction==0.0].index.tolist()
    lab_count[idx,num[1]] += 1
    return lab_count

def ClassPostivePredictions(df):
    print(df[df.prediction==1.0].shape)
    print(df[df.prediction==0.0].shape)

# def parsePoint(line, index2word_set,model, num_features ):
#         print(line)
#         #line = line[3:]
#         featureVec = np.zeros((num_features,),dtype="float32")
#         nwords = 0.0
#         values = [x for x in line.split('`')]
#         #print(values[0])
#         label = values[1].split('|')[1]
#         print(label)
#         if label == '':
#             print(line)
#         text = cleanSent(values[0])
#         #print(" ".join(text))
#         for word in cleanSent(values[0]):
#                 if word and word in index2word_set:
#                 #if word and word not in stop and word in index2word_set:
#                     nwords = nwords + 1.0
#                     featureVec = np.add(featureVec,model[word])
#         featureVec = np.divide(featureVec,nwords)
#         return LabeledPoint(float(label),featureVec)

# def parsePoint1(line):
#     values = [float(x) for x in line.split(' ')]
#     return LabeledPoint(values[0], values[1:])

# def zeromap(line):
#         if line[0].label == 0:
#             line[0].label = float(1)
#         else:
#             line[0].label = float(0.0)
#         return line
#
# def onemap(line):
#         if line[0].label == 1:
#             line[0].label = float(1)
#         else:
#             line[0].label = float(0.0)
#         return line
#
# def twomap(line):
#         if line[0].label == 2:
#             line[0].label = float(1)
#         else:
#             line[0].label = float(0.0)
#         return line
#
# def threemap(line):
#         if line[0].label == 3:
#             line[0].label = float(1)
#         else:
#             line[0].label = float(0.0)
#         return line
#
# def fourmap(line):
#         if line[0].label == 4:
#             line[0].label = float(1)
#         else:
#             line[0].label = float(0.0)
#         return line
#
# def fivemap(line):
#         if line[0].label == 5:
#             line[0].label = float(1)
#         else:
#             line[0].label = float(0.0)
#         return line