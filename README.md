textclassification-naivebayes
=============================

A text classifier using the Naive Bayes Classifier, Maximum Likelihood n-gram Language Model with Laplace add-one smoothing

    class NaiveBayesClassifier
     |  A program to use machine learning techniques with ngram maximum likelihood probabilistic language models as feature to build a bayesian text classifier.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, labels, location)
     |      Constructor method to load training data, and train classifier.
     |  
     |  calculateLikelihood(self, word, label)
     |      Method to calculate Likelihood.
     |  
     |  calculatePrior(self)
     |      Method to calculate Prior.
     |  
     |  classify(self, document)
     |      Method to load test data and classify using training data.
     |  
     |  createUnigram(self)
     |      Method to create Unigram for each class/label.
     |  
     |   getDocuments(self, location)
     |      Method to retrieve test data.
     |  
     |  loadDocuments(self, loc, files)
     |      Method to load labeled data from the training set.
     |  
     |  train(self)
     |      Method to train classifier by calculating Prior and Likelihood.
     |  
     |  unigramProbability(self, word, label)
     |      Method to calculate Unigram Maximum Likelihood Probability with Laplace Add-1 Smoothing.