#!/usr/bin/env python

'''

@author: fries


'''

import argparse
import sys
import string
import re
import glob
import codecs
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn import metrics
import numpy as np
import pdb
import os

def load_corpus(inputdir):
    '''Load document collection'''
    docs = []
    filelist = glob.glob("{}*.txt".format(inputdir))
    for fname in filelist[0:3000]:
        d = []
        with codecs.open(fname,"rU",'utf-8') as fp:
            d += [line.strip().split() for line in fp.readlines()]
        
        d = reduce(lambda x,y:x+y,d)
        docs += [" ".join(d)]
          
    return (docs, filelist)

def load_doc_labels(doclabelfile, filelist):

    labels = []
    doc_labels_dict = {}
    with open(doclabelfile) as FH:
        doc_labels = FH.readlines()
    # Create document-label dictionary
    for entry in doc_labels:
        elements = entry.split(',')
        doc = elements[0]
        label = int(elements[1].rstrip())
        doc_labels_dict[doc] = label
    # Create a list of labels in the same order as the docs
    for path in filelist:
        label = doc_labels_dict[os.path.basename(path)]
        labels.append(label)
    return labels
        
def main(args):
    
    seed = 123456
    
    top_k_ftrs = 15
    corpus = load_corpus(args.inputdir)
    docs = corpus[0]
    filelist = corpus[1]
    #pdb.set_trace()
    
    # load labels here
    labels = load_doc_labels(args.labels, filelist)
    #labels = [int(random.getrandbits(1)) for i in range(len(docs))]
    
    X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=0.33,
                                                        random_state=seed)
   
    print("\n----------------------------")
    print("Logistic Regression")
    print("----------------------------")
    
    lr_clf = LogisticRegression(penalty='l1', C=0.5, class_weight="balanced")
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.50, lowercase=True)
   
    #pdb.set_trace()
    classifier = Pipeline([
                           ('vect', vectorizer),
                           ('clf', lr_clf)])

    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    print metrics.f1_score(y_test,y_pred,pos_label=1)   
    print(metrics.classification_report(y_test, y_pred))
    
    
    coef = lr_clf.coef_.flatten()
    features = []
    ridx = {idx:term for term,idx in vectorizer.vocabulary_.items()}
    for i in range(coef.shape[0]):
        if coef[i] > 0:
            features += [(coef[i],ridx[i])]
            
    for ftr in sorted(features,reverse=1)[0:top_k_ftrs]:
        print ftr
    
    
    
    
    print("\n----------------------------")
    print("Random Forest")
    print("----------------------------")
    
    forest = RandomForestClassifier(n_estimators=100, random_state=0,
                                  class_weight="balanced")
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.50, lowercase=True)
    
    classifier = Pipeline([
                           ('vect', vectorizer),
                           ('forest', forest)])
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
     
    print(metrics.classification_report(y_test, y_pred))
    print "F1-score", metrics.f1_score(y_test,y_pred,pos_label=1)
    
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature ranking:")
    ridx = {idx:term for term,idx in vectorizer.vocabulary_.items()}
    for f in range(top_k_ftrs):
        print("| %d | %s | %f |" % (f + 1, ridx[indices[f]], importances[indices[f]]))
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--documents", type=str, help="input corpus", dest="documents")
    parser.add_argument("-l","--labels", type=str, help="input labels", dest="labels" )
    args = parser.parse_args() 
    
    #args.inputdir = "/Users/fries/Desktop/badge_project/badges_parsed/cleaned/"
    args.inputdir = "/srv/gsfs0/scratch/Redaptor_Psych/badges_parsed/cleaned/"
    
    main(args)
