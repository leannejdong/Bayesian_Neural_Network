
#!/usr/bin/python

# ref: http://iamtrask.github.io/2015/07/12/basic-python-network/  
 

#Sigmoid units used in hidden and output layer. gradient descent and stocastic gradient descent functions implemented with momentum. Note:
#  Classical momentum:

#vW(t+1) = momentum.*Vw(t) - scaling .* gradient_F( W(t) )
#W(t+1) = W(t) + vW(t+1)

#W Nesterov momentum is this: http://cs231n.github.io/neural-networks-3/

#vW(t+1) = momentum.*Vw(t) - scaling .* gradient_F( W(t) + momentum.*vW(t) )
#W(t+1) = W(t) + vW(t+1)

#http://jmlr.org/proceedings/papers/v28/sutskever13.pdf
 

# Numpy used: http://cs231n.github.io/python-numpy-tutorial/#numpy-arrays
 

 

import matplotlib.pyplot as plt
import numpy as np 
import random
import time

#An example of a class
class Network:

    def __init__(self, Topo, Train, Test, MaxTime, Samples, MinPer): 
        self.Top  = Topo  # NN topology [input, hidden, hidden1, output]
        print(self.Top,'Top')
        self.Max = MaxTime # max epocs
        self.TrainData = Train
        self.TestData = Test
        self.NumSamples = Samples

        self.lrate  = 0 # will be updated later with BP call

        self.momenRate = 0
        self.useNesterovMomen = 0 #use nestmomentum 1, not use is 0

        self.minPerf = MinPer
                                        #initialize weights ( W1 W2 W3) and bias ( b1 b2 b3) of the network
        np.random.seed() 
        self.W1 = np.random.randn(self.Top[0]  , self.Top[1])  / np.sqrt(self.Top[0] ) 
        self.B1 = np.random.randn(1  , self.Top[1])  / np.sqrt(self.Top[1] ) # bias first layer
        self.BestB1 = self.B1
        print(self.BestB1, 'B1')
        self.BestW1 = self.W1 
        print(self.BestW1,'W1')
        self.W2 = np.random.randn(self.Top[1] , self.Top[2]) / np.sqrt(self.Top[1] )
        self.B2 = np.random.randn(1  , self.Top[2])  / np.sqrt(self.Top[1] ) # bias second layer
        self.BestB2 = self.B2
        print(self.BestB2,'B2')

        self.BestW2 = self.W2 
        print(self.BestW2,'W2')
        self.W3 = np.random.randn(self.Top[2] , self.Top[3]) / np.sqrt(self.Top[2] )
        self.B3 = np.random.randn(1  , self.Top[3])  / np.sqrt(self.Top[2] ) # bias second layer
        self.BestB3 = self.B3
        print(self.BestB3,'B3')
        self.BestW3 = self.W3 
        print(self.BestW3,'W3')
        self.hidout1 = np.zeros((1, self.Top[1] )) # output of first hidden layer
        print(self.hidout1)
        self.hidout2 = np.zeros((1, self.Top[2] )) # output of 2nd hidden layer
        print(self.hidout2)
        self.out = np.zeros((1, self.Top[3])) #  output last layer

  
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self,actualout):
        error = np.subtract(self.out, actualout)
        sqerror= np.sum(np.square(error))/self.Top[2] 
        #print sqerror
        return sqerror
  
    def ForwardPass(self, X ): 
         z1 = X.dot(self.W1) - self.B1  
         self.hidout1 = self.sigmoid(z1) # output of first hidden layer   
         z2 = self.hidout1.dot(self.W2)  - self.B2 
         self.hitout2 = self.sigmoid(z2)  # output second hidden layer
         print(self.hitout2, 'selfhitout2')
         z3 = self.hidout2.dot(self.W3) - self.B3
         self.out = self.sigmoid(z3)
         print(self.out,'selfout')
         print(X,'X')
    # test forward pass
    #self.W1[0,0]=1.1
    #self.W1[0,1]=0.1
  
    def BackwardPassMomentum(self, Input, desired, vanilla):   
            out_delta = (desired - self.out)*(self.out*(1-self.out))  
            print(out_delta,'outdel')
            hid_delta2 = out_delta.dot(self.W3.T) * (self.hidout2 * (1-self.hidout2))
            print(hid_delta2,'hidel2')
            hid_delta1 = hid_delta2.dot(self.W2.T) * (self.hidout1 * (1-self.hidout1))
            
            if vanilla == 1: #no momentum 
                self.W3+= (self.hidout2.T.dot(out_delta)*self.lrate)
                self.B3+= (-1*self.lrate * out_delta)
                self.W2+= (self.hidout1.T.dot(hid_delta2) * self.lrate)  
                self.B2+=  (-1 * self.lrate * hid_delta2)
                self.W1 += (Input.T.dot(hid_delta1) * self.lrate) 
                self.B1+=  (-1 * self.lrate * hid_delta1)
              
            else:
                v3 = self.W3
                v2 = self.W2 #save previous weights http://cs231n.github.io/neural-networks-3/#sgd
                v1 = self.W1 
                b3 = self.B3
                b2 = self.B2
                b1 = self.B1 
                v3 = ( v3 * self.momenRate) + (self.hidout2.T.dot(out_delta) * self.lrate)
                v2 = ( v2 *self.momenRate) + (self.hidout1.T.dot(hid_delta2) * self.lrate)       # velocity update
                v1 = ( v1 *self.momenRate) + (Input.T.dot(hid_delta1) * self.lrate)   
                v3 = ( v3 *self.momenRate) + (-1 * self.lrate * out_delta)  
                v2 = ( v2 *self.momenRate) + (-1 * self.lrate * hid_delta2)       # velocity update
                v1 = ( v1 *self.momenRate) + (-1 * self.lrate * hid_delta1)   

                if self.useNesterovMomen == 0: # use classical momentum 
                   self.W3+= v3
                   self.W2+= v2
                   self.W1 += v1 
                   self.B3+= b3
                   self.B2+= b2
                   self.B1 += b1 

                else: # useNesterovMomen http://cs231n.github.io/neural-networks-3/#sgd
                   v3_prev = v3
                   v2_prev = v2
                   v1_prev = v1 
                   self.W3 += (self.momenRate * v3_prev + (1 + self.momenRate) )  * v3
                   self.W2 += (self.momenRate * v2_prev + (1 + self.momenRate) )  * v2
                   self.W1 += ( self.momenRate * v1_prev + (1 + self.momenRate) )  * v1 
           

    def TestNetwork(self, Data, testSize, erTolerance):
        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[3])) 
        nOutput = np.zeros((1, self.Top[3]))
        clasPerf = 0
        sse = 0  
        self.W1 = self.BestW1
        self.W2 = self.BestW2 
        self.W3 = self.BestW3 #load best knowledge
        self.B1 = self.BestB1
        self.B2 = self.BestB2 
        self.B3 = self.BestB3 #load best knowledge
     
        for s in xrange(0, testSize):
                
              Input[:]  =   Data[s,0:self.Top[0]] 
              Desired[:] =  Data[s,self.Top[0]:] 
             
              self.ForwardPass(Input ) 
              sse = sse+ self.sampleEr(Desired)  


              if(np.isclose(self.out, Desired, atol=erTolerance).any()):
                 clasPerf =  clasPerf +1  

        return ( sse/testSize, float(clasPerf)/testSize * 100 )

 
    def saveKnowledge(self):
        self.BestW1 = self.W1
        self.BestW2 = self.W2
        self.BestW3 = self.W3
        self.BestB1 = self.B1
        self.BestB2 = self.B2
        self.BestB3 = self.B3  
 
    def BP_GD(self, learnRate, mRate,  useNestmomen , stocastic, vanilla): # BP with SGD (Stocastic BP)
        self.lrate = learnRate
        self.momenRate = mRate
        self.useNesterovMomen =  useNestmomen  
     
        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[3])) 
        Er = []#np.zeros((1, self.Max)) 
        epoch = 0
        bestmse = 100
        bestTrain = 0
        #while  epoch < self.Max and bestTrain < self.minPerf :
        while  epoch < 1:
           
            sse = 0
            for s in xrange(0, self.NumSamples):
                 
                if(stocastic):
                   pat = random.randint(0, self.NumSamples-1) 
                else:
                   pat = s 

                Input[:]  =  self.TrainData[pat,0:self.Top[0]]  
                Desired[:] = self.TrainData[pat,self.Top[0]:]  

               
        
                self.ForwardPass(Input )  
                self.BackwardPassMomentum(Input , Desired, vanilla)
                sse = sse+ self.sampleEr(Desired)
             
            mse = np.sqrt(sse/self.NumSamples*self.Top[3])

            if mse < bestmse:
               bestmse = mse
               self.saveKnowledge() 
               (x,bestTrain) = self.TestNetwork(self.TrainData, self.NumSamples, 0.2)
              

            Er = np.append(Er, mse)
 

            epoch=epoch+1  

        return (Er,bestmse, bestTrain, epoch) 



def normalisedata(data, inputsize, outsize): # normalise the data between [0,1]
    traindt = data[:,np.array(range(0,inputsize))]  
    dt = np.amax(traindt, axis=0)
    tds = abs(traindt/dt) 
    return np.concatenate(( tds[:,range(0,inputsize)], data[:,range(inputsize,inputsize+outsize)]), axis=1)

def main(): 
          
    
        problem = 2 # [1,2,3] choose your problem (Iris classfication or 4-bit parity or XOR gate)
        

        if problem == 1:
           TrDat  = np.loadtxt("train.csv", delimiter=',') #  Iris classification problem (UCI dataset)
           TesDat  = np.loadtxt("test.csv", delimiter=',') #  
           Hidden1 = 6
           Hidden2 = 6
           Input = 4
           Output = 2
           TrSamples =  110
           TestSize = 40
           learnRate = 0.1 
           mRate = 0.01   
           TrainData  = normalisedata(TrDat, Input, Output) 
           TestData  = normalisedata(TesDat, Input, Output)
           MaxTime = 500


           

        if problem == 2:
           TrainData = np.loadtxt("4bit.csv", delimiter=',') #  4-bit parity problem
           TestData = np.loadtxt("4bit.csv", delimiter=',') #  
           Hidden1 = 3
           Hidden2 = 3
           Input = 4
           Output = 1
           TrSamples =  16
           TestSize = 16
           learnRate = 0.9 
           mRate = 0.01
           MaxTime = 3000

        if problem == 3:
           TrainData = np.loadtxt("xor.csv", delimiter=',') #  4-bit parity problem
           TestData = np.loadtxt("xor.csv", delimiter=',') #  
           Hidden = 3
           Input = 2
           Output = 1
           TrSamples =  4
           TestSize = 4
           learnRate = 0.9 
           mRate = 0.01
           MaxTime = 500 

        #print(TrainData)

 
    

        Topo = [Input, Hidden1, Hidden2, Output] 
        MaxRun = 1  # number of experimental runs 
         
        MinCriteria = 95 #stop when learn 95 percent
        
        trainTolerance = 0.2 # [eg 0.15 would be seen as 0] [ 0.81 would be seen as 1]
        testTolerance = 0.4
        
        useStocasticGD = 1 # 0 for vanilla BP. 1 for Stocastic BP
        useVanilla = 1 # 1 for Vanilla Gradient Descent, 0 for Gradient Descent with momentum (either regular momentum or nesterov momen) 
        useNestmomen = 0 # 0 for regular momentum, 1 for Nesterov momentum
         
        

        trainPerf = np.zeros(MaxRun)
        testPerf =  np.zeros(MaxRun)

        trainMSE =  np.zeros(MaxRun)
        testMSE =  np.zeros(MaxRun)
        Epochs =  np.zeros(MaxRun)
        Time =  np.zeros(MaxRun)

        for run in xrange(0, MaxRun  ): 
                 print run
                 fnnSGD = Network(Topo, TrainData, TestData, MaxTime, TrSamples, MinCriteria) # Stocastic GD
                 start_time=time.time()
                 (erEp,  trainMSE[run] , trainPerf[run] , Epochs[run]) = fnnSGD.BP_GD(learnRate, mRate, useNestmomen,  useStocasticGD, useVanilla)   

                 Time[run]  =time.time()-start_time
                 (testMSE[run], testPerf[run]) = fnnSGD.TestNetwork(TestData, TestSize, testTolerance)
                
       	print('trainPerf',trainPerf)
        print('testPerf',testPerf)
        print('trainMSE',trainMSE)
        print('testMSE',testMSE)

        print('Epochs',Epochs)
        print('Time',Time)
        print(np.mean(trainPerf), np.std(trainPerf))
        print(np.mean(testPerf), np.std(testPerf))
        print(np.mean(Time), np.std(Time))
  
  
         
        plt.figure()
        plt.plot(erEp )
        plt.ylabel('error')  
        plt.savefig('out.png')
       
 
if __name__ == "__main__": main()

