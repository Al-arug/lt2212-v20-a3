import os
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
import nltk
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
import csv






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    
    authors = glob("{}/*".format(args.inputdir)) 
    folders = [glob("{}/*".format(i)) for i in authors]
   
   

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    
    
    def tables(author,dims,test_size):
        
        files=[]
        for filename in author:
            list1=""
            with open(filename, "r") as thefile:
                 for line in thefile:
                        list1+=line
                 files.append(list1)
                
        def my_tokenizer(doc):
            tokenized= word_tokenize(doc)
            lem= WordNetLemmatizer()
            return[lem.lemmatize(word,"v") for word in tokenized if word.isalpha()]
    
        vectorizer = CountVectorizer(tokenizer=my_tokenizer)
        X = vectorizer.fit_transform(files)        
 
        
        
        svd = SVD(n_components=dims)
        reduced=svd.fit_transform(X)  
        
        a =len(reduced)*args.testsize//100
      
        
        test= reduced[:a]
        train = reduced[a:]

        
      
        train_labeled = pd.DataFrame(data=train, index = ["train"]*len(train))
        test_labeled = pd.DataFrame(data=test, index = ["test"]*len(test))
       
        
        f = pd.concat([train_labeled, test_labeled],axis=0)
        f.insert(0,"author",[author[0].split("/")[1]]*len(f)) 
            
        return f
    
    
   
    
    print("Writing to {}...".format(args.outputfile))
    
    df = [tables(i,args.dims,args.testsize) for i in folders]
 
    
    p = pd.concat(df, axis=0,sort=False)
 
    p.to_csv(args.outputfile)
       
        

    print("Done!")
    
