import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim 
import random
from torch.autograd import Variable



class Perceptron(nn.Module):
    
    def __init__(self,hidden_layer,nonliniarity,size):
        

        super(Perceptron,self).__init__()
        
        nonliniar={"relu":nn.ReLU(),"softmax":nn.Softmax()}
        self.hidden_layer=hidden_layer
        self.nonliniarity=nonliniarity
        self.size=size
        if self.hidden_layer:
            if self.nonliniarity:
                self.fcl = nn.Linear(size,hidden_layer)
                self.nonLin = nonliniar[nonliniarity]
                self.output = nn.Linear(hidden_layer, 1)
            else:
                self.fcl = nn.Linear(size,hidden_layer)
                self.out = nn.Linear(hidden_layer, 1)
                
        else: 
            self.fc2 = nn.Linear(size,1)
            
        
        
    def forward(self,x):
        if self.hidden_layer:
            if self.nonliniarity: 
                x = self.fcl(x)
                x = self.nonLin(x)
                x = self.output(x)
            else:
                x = self.fcl(x)
                x = self.out(x)
        else: 
            x = self.fc2(x) 
           
        return torch.sigmoid(x)
       
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("hiddenlayer", type=int, default=None,nargs='?', help="If hidden layer add the size")
    parser.add_argument("nonliniarity", type=str, default=None,nargs='?', help="If hidden layer add nonliniar finction by typing either relu or  softmax")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    
    f= pd.read_csv(args.featurefile)
 
    
    test=f[f["Unnamed: 0"]=="test"]
    train=f[f["Unnamed: 0"]=="train"]
    train=train.iloc[:,1:]


    def sampling(df,sample_size):

        
        size=sample_size//2
        samples_1=[]
        samples_0=[]
              
        while True:
            l=random.randint(0,len(df)-1)
            t=random.randint(0,len(df)-1)
            if df.iloc[l,0]==df.iloc[t,0]:
                x = torch.cat((torch.FloatTensor(df.iloc[l,1:]),torch.FloatTensor(df.iloc[t,1:])),0)
                y = torch.FloatTensor([1])
                samples_1.append((x,y))
                if len(samples_1)== size:
                    break
         
        while True:
            l=random.randint(0,len(df)-1)
            t=random.randint(0,len(df)-1)
            if df.iloc[l,0]!=df.iloc[t,0]:
                x = torch.cat((torch.FloatTensor(df.iloc[l,1:]),torch.FloatTensor(df.iloc[t,1:])),0)
                y = torch.FloatTensor([0])
                samples_0.append((x,y))
                if len(samples_0)== size:
                    break
        b= samples_0 + samples_1  
                           
        return random.sample(b,len(b))
    
    train_set = sampling(train,1000)
    size=len(train_set[0][0])
    

    model = Perceptron(args.hiddenlayer,args.nonliniarity,size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)
 

    epochs= 3 
    for epoch in range(epochs):
        for x,y in train_set:
            model.zero_grad()
            x = Variable(x)          
            output= model(x)
            loss= criterion(output,y)
            loss.backward()
            optimizer.step()
            
            
    test = test.iloc[:,1:]      
    test_set = sampling(test,300)
    
                    
    def pred(x):
        if x>0.5:
               b= 1
        else:
               b= 0
        return b
           
    correct=0
    total=0
    true_positive=0
    false_positive=0
    true_negative=0
    false_negative=0
    with torch.no_grad():
        for x,y in test_set:
            output= model(x)
            for idx,i in enumerate(output):
                b= pred(i)
                if b == y[idx]:
                    correct+=1
                if b==1 and y[idx]==1:
                    true_positive+=1
                elif b==1 and y[idx]==0:
                    false_positive+=1
                elif b==0 and y[idx]==0:
                    true_negative+=1
                elif b==0 and y[idx]==1:
                    false_negative+=1
                total+=1
                
    print("accuracy : ", round(correct/total,3))
    precesion = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)
    print("percesion :", precesion)
    print("recall :", recall)
    print("F1 score : ",  2*((precesion*recall)/(precesion+recall)))
                


