#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re as re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

positiveTweetsLabel=1
neutralTweetsLabel=0
negativeTweetsLabel=-1

stopWords = pd.read_csv('/home/abu-bakr/Documents/ML@LUMS/Assignment1/stop_words.txt', header=None)
appleSentimentTweetsData = pd.read_csv('/home/abu-bakr/Documents/ML@LUMS/Assignment1/AppleSentimentTweets.csv')
tweets = appleSentimentTweetsData['text']
sentiments = appleSentimentTweetsData['sentiment']
trainTweetsCount = int(len(tweets) * 0.8)
trainGoldLabels = sentiments[:trainTweetsCount].values
testGoldLabels = sentiments[trainTweetsCount:len(tweets)].values

processedTweets = []
vocabulary = []
trainFeatureMatrix = []
testFeatureMatrix = []

#----PreProcessing----#
def preprocessTweets():
    global tweets
    global processedTweets    
    for tweet in tweets:
        tweet = tweet.lower()
        tweet = cleanStopWordsFromTweet(tweet)
        processedTweets += [' '.join(re.findall('(?<![@a-zA-Z0-9])[A-Za-z]+', re.sub('http[s]*://[a-zA-Z0-9_.%/-]+','', tweet)))]

        
def cleanStopWordsFromTweet(tweet):
    for stopword in stopWords.iloc[:,0]:
        tweet = re.sub(rf'\b{stopword}\b', '', tweet)
    return tweet
#----PreProcessing----#

#----FeatureExtraction----#
def extractFeatures():
    generateVocabulary()
    initializeFeatureMatrices()
    
def generateVocabulary():
    #vocabulary consists of words only from train data####
    global vocabulary
    trainWords = {word for tweet in processedTweets[:trainTweetsCount-1] for word in tweet.split()}
    vocabulary = list(trainWords)

def initializeFeatureMatrices():
    global trainFeatureMatrix
    global testFeatureMatrix
    
    #trainFeatureMatrix with dimensions (training tweets * total vocabulary words)####
    trainFeatureMatrix = np.zeros((trainTweetsCount, len(vocabulary)), dtype = 'int8')
    #testFeatureMatrix with dimensions (totaltweets - trainingtweets * total vocabulary words)####
    testFeatureMatrix = np.zeros((len(tweets) - trainTweetsCount, len(vocabulary)), dtype = 'int8')

    populateFeatureMatrix(trainFeatureMatrix, 0, trainTweetsCount-1)
    populateFeatureMatrix(testFeatureMatrix, trainTweetsCount, len(processedTweets))
    
def populateFeatureMatrix(featureMatrix, startIndex, endindex):
    count = 0    
    for tweet in processedTweets[startIndex:endindex]:
        for word in tweet.split():
            if word in vocabulary:
                featureMatrix[count][vocabulary.index(word)] += 1
        count += 1
#----FeatureExtraction----#


#----KNN Classification----#

def performKNNClassification():

    crossValidationKValues = [1,2,3,4,5,6,7,8,9,10]
    validationDatasetLength = int(trainTweetsCount * 0.2)
    crossValidationFolds = 5
    
    print('Cross-Validation Start Time: ' , datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    euclideanKWiseAccuracyDict = {}
    euclideanKWiseF1scoreDict = {}
    #Cross Validate for Euclidean Distance Metric
    crossValidate(crossValidationKValues, validationDatasetLength, crossValidationFolds, 
                  euclideanKWiseAccuracyDict, euclideanKWiseF1scoreDict, "euclidean")
    
    manhattanKWiseAccuracyDict = {}
    manhattanKWiseF1scoreDict = {}
    #Cross Validate for Manhattan Distance Metric
    crossValidate(crossValidationKValues, validationDatasetLength, crossValidationFolds,
                  manhattanKWiseAccuracyDict, manhattanKWiseF1scoreDict, "manhattan")
    
    print('Cross-Validation End Time: ' , datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    plotAccuracyStats(crossValidationKValues, euclideanKWiseAccuracyDict, manhattanKWiseAccuracyDict)
    plotF1Stats(crossValidationKValues, euclideanKWiseF1scoreDict, manhattanKWiseF1scoreDict)
    
    print('As evident from graphs, Suitable value of K for Euclidean: 2, Suitable value of K for Manhattan: 7..')
    print('\n We will prefer F1 Count here while selecting appropriate value of K.')
    print('This is because data is skewed/imbalanced. Most of tweets are of negative nature.')
    
    print('Doing Testset Classification for Euclidean; K = 2')
    k = 2
    classifyTestData("euclidean", k)
    
    print('Doing Testset Classification for Manhattan; K = 7')
    k = 7
    classifyTestData("manhattan", k)
    
def crossValidate(crossValidationKValues, validationDatasetLength, crossValidationFolds,kWiseAccuracyDict,
                  kWiseF1scoreDict, distanceMetric):
    print('Starting cross-validation with distance metric: ' , distanceMetric)
    
    for k in crossValidationKValues:
        foldCounter = 0
        confusionMatrix = np.zeros(shape=(3,3))
        #Initial range for validation data split (0th row to 20pc/260 rows)
        validationStartIndex = 0
        validationEndIndex = 0 + validationDatasetLength #### initial validationDatasetLength: 20 pc of training data
        while(foldCounter < crossValidationFolds):
            #extracting gold lables for validation range from all training gold labels
            goldLabels = trainGoldLabels[validationStartIndex:validationEndIndex]
            #calculating distances between validation and training sets; result is, for each distance,..
            #..training points indices sorted from lowest distance to highest
            sortedDistanceIndicesMatrix = caculateValidationSetDistances(distanceMetric, validationStartIndex, validationEndIndex)
            #extract k lables from sorted distances matrix
            kPredictedLabels = predictKLabels(sortedDistanceIndicesMatrix, k)
            #predict classes on basis of most frequent class in k lables
            predictedClassLabels = predictClasses(kPredictedLabels, k)
            #generate confusion matrix; compare predicted and gold lables
            #also summing confusion matrix for each fold; end result will be one confusion matrix for each..
            #..value of k and distance metric
            confusionMatrix = confusionMatrix + calculateConfusionMatrix(predictedClassLabels, goldLabels)
            
            #increment validation data range start and end inde
            validationStartIndex = validationEndIndex
            validationEndIndex = validationEndIndex + validationDatasetLength
            
            #increment fold counter
            foldCounter += 1
            
        accuracy = calculateAccuracy(confusionMatrix)
        kWiseAccuracyDict[k] = accuracy
        
        precision = calculatePrecision(confusionMatrix)
        recall = calculateRecall(confusionMatrix)
        
        f1score = calculateF1Score(precision, recall)
        kWiseF1scoreDict[k] = f1score
        
        print('Finished Processing For k = ' , k)
        print('Confusion Matrix: \n', confusionMatrix)
        print('Precision: ', precision)
        print('Accuracy: ', accuracy)
        print('Recall: ', recall)
        print('F1score: ', f1score)
        print('\n')
            
def caculateValidationSetDistances(distanceMetric, validationStartIndex, validationEndIndex):
    #generate validation dataset from training feature matrix
    validationDataset = trainFeatureMatrix[validationStartIndex:validationEndIndex,:]
    #delete(this function not actually deletes but just ignores) validation data in training matrix..
    #..and get remainnig training data
    trainingDataset = np.delete(trainFeatureMatrix, np.s_[validationStartIndex:validationEndIndex], axis = 0)
    
    if distanceMetric == "euclidean":
        distanceMatrix = np.sqrt(np.sum((validationDataset[:,None] - trainingDataset) ** 2, axis = 2))
    elif distanceMetric == "manhattan":
        distanceMatrix = np.sum(abs(validationDataset[:,None] - trainingDataset), axis = 2)
    
    return distanceMatrix.argsort()

def predictKLabels(indicesMatrix, k):
    #return given k labels of training points having shortest distance with validation points
    return trainGoldLabels[indicesMatrix[:,0:k]]

def predictClasses(kPredictedLabels, k):
    #return most frequent class for each training-distance point from klables
    predictedClassLabels = np.empty(len(kPredictedLabels))
    loopIter = 0
    for kLabels in kPredictedLabels:
        kLabelsEndIndex = k
        while(True):
            values, counts = np.unique(kLabels[0:kLabelsEndIndex], return_counts = True)
            if(isTieExists(counts)):
                kLabelsEndIndex -= 1
            else:
                predictedClassLabels[loopIter] = values[np.argmax(counts)]
                loopIter += 1
                break
    return predictedClassLabels

def isTieExists(counts):
    return len(counts) != len(set(counts))
    
def calculateConfusionMatrix(predictedClassLabels, goldLabels):
    confusionMatrix = np.empty(shape=(3,3))
 
    positiveGoldLabelsDetails = getGoldLabelsDetails(predictedClassLabels, goldLabels, positiveTweetsLabel)
    confusionMatrix[0][0] = positiveGoldLabelsDetails.get(positiveTweetsLabel)
    confusionMatrix[0][1] = positiveGoldLabelsDetails.get(neutralTweetsLabel)
    confusionMatrix[0][2] = positiveGoldLabelsDetails.get(negativeTweetsLabel)
    
    neutralGoldLablesDetails = getGoldLabelsDetails(predictedClassLabels, goldLabels, neutralTweetsLabel)
    confusionMatrix[1][0] = neutralGoldLablesDetails.get(positiveTweetsLabel)
    confusionMatrix[1][1] = neutralGoldLablesDetails.get(neutralTweetsLabel)
    confusionMatrix[1][2] = neutralGoldLablesDetails.get(negativeTweetsLabel)
    
    negativeGoldLablesDetails = getGoldLabelsDetails(predictedClassLabels, goldLabels, negativeTweetsLabel)
    confusionMatrix[2][0] = negativeGoldLablesDetails.get(positiveTweetsLabel)
    confusionMatrix[2][1] = negativeGoldLablesDetails.get(neutralTweetsLabel)
    confusionMatrix[2][2] = negativeGoldLablesDetails.get(negativeTweetsLabel)
    
    confusionMatrix[np.isnan(confusionMatrix)] = 0
    return confusionMatrix

def getGoldLabelsDetails(predictedClassLabels, goldLabels, label):
    values, counts = np.unique(predictedClassLabels[np.where(goldLabels==label)[0]], return_counts = True)
    labelsDetails = dict(zip(values, counts))
    return labelsDetails

def calculateAccuracy(confusionMatrix):
    with np.errstate(divide='ignore', invalid='ignore'):
        return checkIsNaN((confusionMatrix[0][0] + confusionMatrix[1][1] + confusionMatrix[2][2]) / np.sum(confusionMatrix))

def calculatePrecision(confusionMatrix):
    with np.errstate(divide='ignore', invalid='ignore'):
        return checkIsNaN((((confusionMatrix[0][0] / np.sum(confusionMatrix[0,:])) + (confusionMatrix[1][1] / np.sum(confusionMatrix[1,:])) + (confusionMatrix[2][2] / np.sum(confusionMatrix[2,:]))) / 3))

def calculateRecall(confusionMatrix):
    with np.errstate(divide='ignore', invalid='ignore'):
        return checkIsNaN((((confusionMatrix[0][0] / np.sum(confusionMatrix[:,0])) + (confusionMatrix[1][1] / np.sum(confusionMatrix[:,1])) + (confusionMatrix[2][2] / np.sum(confusionMatrix[:,1]))) / 3))

def calculateF1Score(precision, recall):
    with np.errstate(divide='ignore', invalid='ignore'):
        return checkIsNaN((2 * precision * recall) / (precision + recall))

def checkIsNaN(value):    
    if np.isnan(value):
        return 0
    else:
        return value
    
def plotAccuracyStats(crossValidationKValues, euclideanKWiseAccuracyDict, manhattanKWiseAccuracyDict):

    y_euclidean = list(euclideanKWiseAccuracyDict.values())
    y_manhattan = list(manhattanKWiseAccuracyDict.values())
    
    plt.subplot(1, 1, 1)
    plt.plot(crossValidationKValues, y_euclidean, color='green')
    plt.plot(crossValidationKValues, y_manhattan, color='orange')
    plt.xticks(np.arange(min(crossValidationKValues), max(crossValidationKValues) + 1, 1))

    plt.xlabel('K-Values')
    plt.ylabel('Accuracy')
    plt.title('K-Accuracy Plot')
    plt.legend(['Euclidean', 'Manhattan'], loc ="upper left")
    plt.show() 
    
def plotF1Stats(crossValidationKValues, euclideanKWiseF1scoreDict, manhattanKWiseF1scoreDict):
    y_euclidean = list(euclideanKWiseF1scoreDict.values())
    y_manhattan = list(manhattanKWiseF1scoreDict.values())
        
    plt.subplot(1, 1, 1)
    plt.plot(crossValidationKValues, y_euclidean, color='green')
    plt.plot(crossValidationKValues, y_manhattan, color='orange')
    plt.xticks(np.arange(min(crossValidationKValues), max(crossValidationKValues) + 1, 1))
    plt.yticks(np.arange(min(y_euclidean), max(y_euclidean) + 0.01, 0.01))
    
    plt.xlabel('K-Values')
    plt.ylabel('F1-Score')
    plt.title('K-F1Score Plot')
    plt.legend(['Euclidean', 'Manhattan'], loc ="upper left")
    plt.show()

def classifyTestData(distanceMetric, k):
    sortedDistanceIndicesMatrix = caculateTestSetDistances(distanceMetric)
    kPredictedLabels = predictKLabels(sortedDistanceIndicesMatrix, k)
    predictedClassLabels = predictClasses(kPredictedLabels, k)
    confusionMatrix = calculateConfusionMatrix(predictedClassLabels, testGoldLabels)
    print('Finished Testset Classification for Distance Metric: ' , distanceMetric)
    print('Confusion Matrix: \n', confusionMatrix)
    
    accuracy = calculateAccuracy(confusionMatrix)
    precision = calculatePrecision(confusionMatrix)
    recall = calculateRecall(confusionMatrix)
    f1score = calculateF1Score(precision, recall)

    print('Precision: ', precision)
    print('Accuracy: ', accuracy)
    print('Recall: ', recall)
    print('F1score: ', f1score)
    print('\n')
    
def caculateTestSetDistances(distanceMetric):    
    if distanceMetric == "euclidean":
        distanceMatrix = np.sqrt(np.sum((testFeatureMatrix[:,None] - trainFeatureMatrix) ** 2, axis = 2))
    elif distanceMetric == "manhattan":
        distanceMatrix = np.sum(abs(testFeatureMatrix[:,None] - trainFeatureMatrix), axis = 2)
    
    return distanceMatrix.argsort()    
#----KNN Classification----#    


preprocessTweets()
extractFeatures()
performKNNClassification()


# In[2]:


####IMPORTANT: Execute cells in order; pre-processed data is required here from step 1####


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics

crossValidationKValues = [1,2,3,4,5,6,7,8,9,10]
euclideanAccuracies = []
euclideanF1Scores = []
manhattanAccuracies = []
manhattanF1Scores = []

def crossValidateUsingScikit(distanceMetric, accuracies, f1scores):
    for k in crossValidationKValues:
        knn = KNeighborsClassifier(n_neighbors = k, metric = distanceMetric)
        predictedSentiments = cross_val_predict(knn, trainFeatureMatrix, trainGoldLabels, cv = 5)
        confusionMatrix = confusion_matrix(trainGoldLabels, predictedSentiments)
        classificationReport = classification_report(trainGoldLabels, predictedSentiments, output_dict=True)
        print('For value of k: ', k, ' and distance metric: ' , distanceMetric)
        print('Confusion Matrix: \n', confusionMatrix)
        print('Accuracy: ', classificationReport.get('accuracy'))
        accuracies.append(classificationReport.get('accuracy'))
        print('Precision: ', classificationReport.get('macro avg').get('precision'))
        print('Recall: ', classificationReport.get('macro avg').get('recall'))
        print('F1Score: ', classificationReport.get('macro avg').get('f1-score'))
        f1scores.append(classificationReport.get('macro avg').get('f1-score'))
        print('\n')

def plotStats(crossValidationKValues, euclideanList, manhattanList, xlabel, ylabel, title):
    plt.subplot(1, 1, 1)
    plt.plot(crossValidationKValues, euclideanList, color='green')
    plt.plot(crossValidationKValues, manhattanList, color='orange')
    plt.xticks(np.arange(min(crossValidationKValues), max(crossValidationKValues) + 1, 1))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(['Euclidean', 'Manhattan'], loc ="upper left")
    plt.show() 

def classifyTestInstancesUsingScikit(distanceMetric, k):
    knn = KNeighborsClassifier(n_neighbors = k, metric = distanceMetric)
    knn.fit(trainFeatureMatrix, trainGoldLabels)
    predictedSentiments = knn.predict(testFeatureMatrix)
    
    classificationReport = classification_report(testGoldLabels, predictedSentiments, output_dict=True)
    print('For Distance Metric: ' , distanceMetric)
    print('Accuracy: ', classificationReport.get('accuracy'))
    print('Precision: ', classificationReport.get('macro avg').get('precision'))
    print('Recall: ', classificationReport.get('macro avg').get('recall'))
    print('F1Score: ', classificationReport.get('macro avg').get('f1-score'))
    print('\n')

crossValidateUsingScikit('euclidean', euclideanAccuracies, euclideanF1Scores)
crossValidateUsingScikit('manhattan', manhattanAccuracies, manhattanF1Scores)

plotStats(crossValidationKValues, euclideanAccuracies, manhattanAccuracies, 'K-Values', 'Accuracy', 'K-Accuracy Plot [Scikit]')
plotStats(crossValidationKValues, euclideanF1Scores, manhattanF1Scores, 'K-Values', 'F1-Scores', 'K-F1Scores Plot [Scikit]')

print('As evident from graphs, Suitable value of K for Euclidean and Manhattan: 2', '\n')
classifyTestInstancesUsingScikit('euclidean', 2)
classifyTestInstancesUsingScikit('manhattan', 2)

