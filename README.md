# LT2212 V20 Assignment 3

Put any documentation here including any answers to the questions in the 
assignment on Canvas.

Part 1: 

a3.features.py contains a function that takes one folder/author at the time. It  vectorizes,reduces dimension, labels the data iwht the author name and split it between train and split. The train and tet size can be determined from terminal at the desired size. For the dimensions it is possible to reduce from 1 to 14 demension not more dimensions. The reason for the dimension limit is that the data of all the folders are concatonated later and some folders when they are reduced they get reuduced to lower dimesnions than the the demension given to SVD. This is probably because of the differences of counting and the frequencies that each folder contains. 

To run the code as an example: python a3_features.py enorn_sample output1.csv 14 -T 20

The concatonated data frames then printed  to the given file name as instructed. 


part 2 & 3 : 

For labeling the data I made a function that takes in a data frame, eaither for test or train, and return a list of tensors in tuples, each tuple contains x and y. x is a tensor of two documents concatenated and y is the label. The loss function is binary cross-entropy. The hidden layer is optional then if hidden layer is chosen you can add nonlinear function or not if not specified. The nonlinear functions are ReLU and softmax. I chose softmax because I thought its probabilistic distribution of the first output might increase accuracy, hence I chose no number of dimensons for the soft max to output but it will output the same number of dimensions as the given input. The second choice is Relu, because it has an effective classification of relatively high gradient up to one where negative numbers are turned to zero. We could see in the comming evalualuation that recall and percession are relatively higher when using Relu which means Relu might have contributed for better classification of true positives and less false negative and false positives. I have done the evaluation by calculating the results manually.

To run the code with no hidden layers nor nonliniarities: python a3_model.py output1.csv 

with a hidden layer of size 100 and no non-liniarity: python a3_model.py output1.csv 100 

with a layer of hundred and relu:  python a3_model.py output1.csv 100 relu 

with  a layer of hundred and softmax: python a3_model.py output1.csv 100 softmax

         hiddenlayer   nonliniarity    accuracy    percesion      recall   f1_score
        
          None           None           0.513      0.515625        0.44     0.4748201438848921
          
          20             None           0.557      0.5586          0.54     0.5491525423728814
          
         100             None           0.567      0.6162          0.353    0.449152542
         
         20            softmax          0.693      0.7013          0.673    0.6870
          
        100            softmax          0.693      0.685           0.713    0.699
        
        20             relu             0.62       0.613            0.646    0.629
        
       100             relu             0.67       0.615            0.90     0.73
       
The scores were much better with hidden layer and non liniarity function, either with soft max or with relu. Bigger hidden layer without non liniarity also improved the preformance but not as good as with non-linearity functions. Note: sometimes the first run produces a very low recall and percesion, but in another run for the code percesion and recall went up again. I thought the results should be kind of close to each others at any time when running the code. Though accuracy was almost similar most of the time      

