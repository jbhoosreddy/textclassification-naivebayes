from __future__ import division
#from ngram import nGram
from collections import Counter
import os, re
import math as calc

class NaiveBayesClassifier():
    """A program to use machine learning techniques with ngram maximum likelihood probabilistic language models as feature to build a bayesian text classifier.
Usage:
>>> tc = NaiveBayesClassifier(['tragedy', 'poem', 'history', 'comedy'], 'documents/')
>>> tc.classify('documents/shakespeare-comedy-merchant.txt')
comedy"""
    def __init__(self, labels, location):
        """Constructor method to load training data, and train classifier."""
        #self.ng = nGram(False, False)
        self.labels = labels
        self.files = self.getDocuments(location)
        self.words = self.loadDocuments(location, self.files)
        self.train()
        return

    def train(self):
        """Method to train classifier by calculating Prior and Likelihood."""
        self.prior = self.calculatePrior()
        self.unigram = self.createUnigram()
        
    def classify(self, document):
        """Method to load test data and classify using training data."""
        file = open(document, 'r')
        words = re.findall(r"<?/?\w+>?",file.read().lower())
        P = dict.fromkeys(self.labels, 0)
        for label in self.labels:
            for word in words:
                P[label] = P[label] + self.calculateLikelihood(word, label)
        P[label] = P[label] + calc.log(self.prior[label])
        print sorted(P, key=P.get, reverse=True)[0]
            
    def getDocuments(self, location):
        """Method to retrieve test data."""
        labels = self.labels
        files = dict.fromkeys(labels, [])
        for label in labels:
            labelfile=[]
            for file in os.listdir(location+label+'/'):
                if label in file:
                    labelfile.append(file)
            files[label] = labelfile
        return files

    def calculatePrior(self):
        """Method to calculate Prior."""
        prior = dict()
        for label in self.labels:
            prior[label] = len(self.files[label])
        s = sum(prior.values())
        for label in prior.keys():
            prior[label] = prior[label]/s
        return prior

    def calculateLikelihood(self, word, label):
        """Method to calculate Likelihood."""
        return self.unigramProbability(word, label)

    def unigramProbability(self, word, label):
        """Method to calculate Unigram Maximum Likelihood Probability with Laplace Add-1 Smoothing."""
        return calc.log((self.unigram[label][word]+1)/(len(self.words[label])+len(self.unigram[label])))

    def createUnigram(self):
        """Method to create Unigram for each class/label."""
        unigram = dict.fromkeys(self.labels, dict())
        for label in self.labels:
            unigram[label] = Counter(self.words[label])
        return unigram
            
    def loadDocuments(self, loc, files):
        """Method to load labeled data from the training set."""
        words = dict.fromkeys(self.labels,"")
        for label in self.labels:
            for file in files[label]:
                handle = open(loc+label+'/'+file, 'r')
                words[label] = words[label] + ' ' + handle.read()
                handle.close()
        for label in self.labels:
            words[label] = re.findall(r"<?/?\w+>?",words[label].lower())
        return words
