import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from tensorflow.keras import layers

from keras import backend as k
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils.vis_utils import plot_model
from numpy.random import randint
from random import choice
from numpy.random import uniform

import matplotlib.pyplot


#lecture du dataset
def kddnsl(show_examples=False):
    dftrain = pd.read_csv('KDDNSLFULL.csv')
    dftest=pd.read_csv('KDDTEST.csv')

    #print(dftrain.head())
    #print(dftest.head())


    rows, cols = dftrain.shape
    rowst, colst = dftest.shape

    label_names = dftrain['attacktype'].unique()
    label_namest = dftest['attacktype'].unique()



    index_and_label = list(enumerate(label_names))
    index_and_labelt = list(enumerate(label_namest))

    label_to_index = dict((label, index) for index, label in index_and_label)
    label_to_indext = dict((label, index) for index, label in index_and_labelt)


    dftrain = dftrain.replace(label_to_index)
    dftest = dftest.replace(label_to_indext)

    dftrain = dftrain.sample(frac=1.0)
    dftest = dftest.sample(frac=1.0)

    train_data = dftrain
    test_data = dftest

    x_train = train_data.iloc[:,:-1]
    y_train = train_data.iloc[:, -1:]
    #print (len(x_train))
    y_train = to_categorical(y_train, 5)[:len(x_train)]
    #print(x_train.shape)
    #print(y_train.shape)

    x_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:, -1:]
    y_test = to_categorical(y_test, 5)[:len(x_train)]
    #print(x_test.shape)
    #print(y_test.shape)
    i_r, i_c, n_c = 28, 28, 5
    i_sh = (1, i_r, i_c)
    #print(i_sh)
    return x_train, x_test, y_train, y_test,i_sh


#classe permettant de crÃ©er le rÃ©seau de neurones
class Net:
    def __init__(self,epoch,dropout,hiddenActivationFn,outputActivationFn,lossFunction,optimiser,totalAttributes,totalHiddenLayers,neuronesPerLayer):
      self.epoch = epoch #epoch value
      self.dropout =dropout #drop out value
      self.hiddenActivationFn = hiddenActivationFn #activation for hidden layer
      self.outputActivationFn=outputActivationFn    #activation function for the output layer
      self.lossFunction = lossFunction #loss function
      self.optimiser = optimiser #optimization
      self.accuracy = 0 #accuracy
      self.totalAttributes = totalAttributes #total attributes/features of the dataset [n]
      self.totalHiddenLayers = totalHiddenLayers #total hidden layers per neural network [pMax]
      self.neuronesPerLayer = neuronesPerLayer #total neurones per layer
      self.parentDropOut = 0.0 #total dropout per parent
      self.parentTotalHiddentLayer =  0#self.parentTotalHiddentLayer #total hiddentlayers per parent
      self.patentTotalNeurone = 0#self.patentTotalNeurone #total neurone per parent


    def init_params(self):
        params = { 'epochs'                 : self.epoch,
                   'dropout'                : self.dropout,
                  'hiddenActivation'        : self.hiddenActivationFn,
                  'outputActivation'        : self.outputActivationFn, 
                  'loss'                    : self.lossFunction,
                  'optimizer'               : self.optimiser,
                  'accuracy'                : self.accuracy,
                  'totalAttributes'         : self.totalAttributes,
                  'totalHiddenLayers'       : self.totalHiddenLayers,
                  'neuronesPerLayer'        : self.neuronesPerLayer,
                  'parentDropOut'           : self.parentDropOut,  #total dropout per parent
                  'parentTotalHiddentLayer' : self.parentTotalHiddentLayer, #total hiddentlayers per parent
                  'patentTotalNeurone'      : self.patentTotalNeurone #total neurone per parent
                }
        return params

def init_net(epoch,dropout,hiddenActivationFn,outputActivationFn,lossFunction,optimiser,totalAttributes,totalHiddenLayers,neuronesPerLayer,p):
    return [Net(epoch,dropout,hiddenActivationFn,outputActivationFn,lossFunction,optimiser,totalAttributes,totalHiddenLayers,neuronesPerLayer) for _ in range(p)]

def fitness(n, n_c, i_shape, x, y, b, x_test, y_test,pMax):
  for cnt, i in enumerate(n):
      p = i.init_params()

      epoch               = p['epochs'] #epoch value
      dropout             = p['dropout'] #drop out value
      hiddenActivationFn  = p['hiddenActivation'] #activation for hidden layer
      outputActivationFn  = p['outputActivation']    #activation function for the output layer
      lossFunction        = p['loss'] #loss function
      optimiser           = p['optimizer'] #optimization
      totalAttributes     = p['totalAttributes'] #total attributes/features of the dataset [n]
      totalHiddenLayers   = p['totalHiddenLayers'] #total hidden layers per neural network [pMax]
      neuronesPerLayer    = p['neuronesPerLayer'] #total neurones per layer
      parentDropOut       = p['parentDropOut']  #total dropout per parent
      parentTotalHiddentLayer   =  p['parentTotalHiddentLayer'] #total hiddentlayers per parent
      patentTotalNeurone =p['patentTotalNeurone']
       #total neurone per parent
      try:
        # Parameter name    # Suggested value
          m = net_model(epoch               = epoch,            # epoch number   
                        hiddenActivationFn  = hiddenActivationFn,            # hiden layers activation function
                        outputActivationFn  = outputActivationFn,            # output layer activation function
                        dropout             = dropout,                       # dropout 2        Not used by algorithm
                        optimiser           = optimiser,                     # optimizer               'adadelta'
                        lossFunction        = lossFunction,                  # loss function           'categorical crossentropy'
                        xtrain              = x,              # train data
                        ytrain              = y,              # train label
                        batchSize           = b,              # bias value
                        xtest               = x_test,    # test data
                        ytest               = y_test,    # test label
                        totalAttributes     = totalAttributes,
                        totalHiddenLayers   = totalHiddenLayers,
                        neuronesPerLayer    = neuronesPerLayer,
                        pMax                = pmax,
                        parentDropOut       = parentDropOut,  #total dropout per parent
                        parentTotalHiddentLayer   =  parentTotalHiddentLayer, #total hiddentlayers per parent
                        patentTotalNeurone =patentTotalNeurone
                        )    

          # # Current best: 99.15%

          s = m.evaluate(x=x_test, y=y_test, verbose=0)
          i.accuracy = s[1]

          #print(m.summary())
          modelConfig = m.get_config()
          #print(modelConfig['layers'])
          #print(type(modelConfig['layers'])
          print( modelConfig['config'])
          #print("total layer "+ str(len(modelConfig['layers'])))
          count = 0
          patentTotalNeurone = 0
          parentTotalHiddentLayer = 0
          parentDropOut=0

          print ('-----------------------------------------------FONCTION DE FITNESS-----------------------------------------------')
          totalList = len(modelConfig['layers'])-1
          #print("Total Number of "+ str(totalList))
          for layer in modelConfig['layers']:
            while (count < len(modelConfig['layers'])-1):
              #print(str(count) + layer['class_name'] )
              count = count+1
            
            #Exclue les couches ayant les fonctions d'activations et la couche d'entrée
            if ((layer['class_name'] != 'Activation') and (layer['class_name'] != 'InputLayer')):
              if (count>2): #exclus la couche entrée
                if(layer['class_name']== 'Dense'): # les couches cachées
                  patentTotalNeurone = patentTotalNeurone + layer['config']['units']
                  print ("Nombre de neurone du parent: " + str(patentTotalNeurone))
                  print('------------------------------------------------------------------')

                  parentTotalHiddentLayer = parentTotalHiddentLayer + 1
                  print ("Nombre de couche cachée du parent: " + str(parentTotalHiddentLayer))
                  print('------------------------------------------------------------------')

                if(layer['class_name']== 'Dropout'):
                  parentDropOut = parentDropOut+layer['config']['rate']
                  print  ("Drop out du parent:" + str(parentDropOut))
                print('------------------------------------------------------------------')
                
          #print(m.get_config())
          print('Accuracy: {}'.format(round((i.accuracy * 100),2)))
          print("test test")
          print(p)
          print ('-----------------------------------------------FONCTION DE FITNESS-----------------------------------------------')
      except Exception as e:
          print(e)
  return n

#fonction qui gÃ©nÃ¨re le rÃ©seau de neurones
def net_model(epoch, hiddenActivationFn,outputActivationFn, dropout, optimiser, lossFunction,  xtrain, ytrain, batchSize, xtest, ytest,totalAttributes,totalHiddenLayers,neuronesPerLayer,pMax, parentDropOut,parentTotalHiddentLayer,patentTotalNeurone):
    model = Sequential()

    print("Hidden layers will be "+  str(totalHiddenLayers))
    print ('-------------------------------------------GENERATION DU RESEAU DE NEURONE----------------------------------------------')
    inputNeurones = randint(totalAttributes/2,totalAttributes)
    print('Nombre de neurones de la couche d''entrée:' + str(neuronesPerLayer))
    model.add(Dense(inputNeurones, input_shape=[totalAttributes,])) 
    model.add(Activation(hiddenActivationFn))

    #hidden layers
    for i in range(totalHiddenLayers):
      HiddenNeurones = randint(totalAttributes/2,totalAttributes)
      print('Couche Cachée: ' + str(i))
      print('------------------------------------------------------------------')
      print('------------------------------------------------------------------')
      print('Nombre de neurone de la couche: ' + str(HiddenNeurones))
      dropoutVal = random.uniform(0,1)
      print("Drop out de la couche: " + str(dropoutVal))
      print('------------------------------------------------------------------')
      model.add(Dense(HiddenNeurones))
      model.add(Activation(hiddenActivationFn))
      model.add(layer=Dropout(rate=dropoutVal))
      print('Couche Cachée: ' + str(i))
      print('------------------------------------------------------------------')
      print('------------------------------------------------------------------')
    #output layers
    model.add(Dense(5))
    model.add(Activation(outputActivationFn))
    model.compile(optimizer=optimiser, loss=lossFunction, metrics=['accuracy'])
    model.fit(x=xtrain, y=ytrain, batch_size=batchSize, epochs=epoch, verbose=0, validation_data=(xtest, ytest))
   # print(model.summary())
    tf.keras.utils.plot_model(model, to_file='model_plot3.png', show_shapes=True, show_layer_names=True,expand_nested=True)
    print ('-------------------------------------------GENERATION DU RESEAU DE NEURONE----------------------------------------------')
    return model

def selection(n):
  n = sorted(n, key=lambda j: j.accuracy, reverse=True)
  n = n[:int(len(n))]

  return n

def crossover(n,pMax,maxNeurone):
    offspring = []
    p1 = choice(n)
    p2 = choice(n)
    print ('---------------------------------------------------CROSSOVER----------------------------------------------------')
    #print('Nombre de neurone du parent P1: '+ str(p1.patentTotalNeurone) + ' et  P2:' + str(p2.patentTotalNeurone))
    print('Nombre de neurone du parent P1: '+ str(p1.neuronesPerLayer) + ' et  P2:' + str(p2.neuronesPerLayer))
    print('------------------------------------------------------------------')
   # print('Drop out du parent P1:'+ str(p1.parentDropOut) + ' et P2:' + str(p2.parentDropOut))
    print('Drop out du parent P1:'+ str(p1.dropout) + ' et P2:' + str(p2.dropout))
    print('------------------------------------------------------------------')

    """if (round((p1.patentTotalNeurone + p2.patentTotalNeurone)/2) >maxNeurone):
      neuronesPerLayerChild = round((p1.patentTotalNeurone + p2.patentTotalNeurone)/4) 
    else:
      neuronesPerLayerChild = round((p1.patentTotalNeurone + p2.patentTotalNeurone)/2) #moyenne du nombre de neuronnes des 2 parents
    
    if (round((p1.parentDropOut + p2.parentDropOut)/2) >1):
      dropoutofChild = round((p1.parentDropOut + p2.parentDropOut)/4) #moyenne du nombre de neuronnes des 2 parents
    else:
      dropoutofChild = round((p1.parentDropOut + p2.parentDropOut)/2)

    if(round((p1.parentTotalHiddentLayer + p2.parentTotalHiddentLayer)/2) >pMax):
        numberofHiddenLayersPerChild = round((p1.parentTotalHiddentLayer+p2.parentTotalHiddentLayer)/4) #nombre de couche cachÃ©es
    elif (round((p1.parentTotalHiddentLayer + p2.parentTotalHiddentLayer)/2) <pMax):
      #ajout de nouvelle couche au hasard ayant un drop out de 0
      numberofHiddenLayersPerChild = round((p1.parentTotalHiddentLayer+p2.parentTotalHiddentLayer)/2) 
    else:
        numberofHiddenLayersPerChild = round((p1.parentTotalHiddentLayer+p2.parentTotalHiddentLayer)/2) #nombre de couche cachÃ©es
    """
    if (round((p1.neuronesPerLayer + p2.neuronesPerLayer)/2) >maxNeurone):
      neuronesPerLayerChild = round((p1.neuronesPerLayer + p2.neuronesPerLayer)/4) 
    else:
      neuronesPerLayerChild = round((p1.neuronesPerLayer + p2.neuronesPerLayer)/2) #moyenne du nombre de neuronnes des 2 parents
    
    if (round((p1.dropout + p2.dropout)/2) >1):
      dropoutofChild = round((p1.dropout + p2.dropout)/4) #moyenne du nombre de neuronnes des 2 parents
    else:
      dropoutofChild = round((p1.dropout + p2.dropout)/2)

    if(round((p1.totalHiddenLayers + p2.totalHiddenLayers)/2) >pMax):
        numberofHiddenLayersPerChild = round((p1.totalHiddenLayers+p2.totalHiddenLayers)/4) #nombre de couche cachÃ©es
    elif (round((p1.totalHiddenLayers + p2.totalHiddenLayers)/2) <pMax):
      #ajout de nouvelle couche au hasard ayant un drop out de 0
      numberofHiddenLayersPerChild = round((p1.totalHiddenLayers+p2.totalHiddenLayers)/2) 
    else:
        numberofHiddenLayersPerChild = round((p1.totalHiddenLayers+p2.totalHiddenLayers)/2) #nombre de couche cachÃ©es
    #!!!!!!!!! TO DO !!!!!!!!!
    # To loop throught the Neural Network and get the different parameters
    # if numberofHiddenLayersPerChild < Pmax, insert new random layers in P1 and P2 with drop out value = 0 => In Progress

    c1 = Net(p1.epoch, dropoutofChild, p1.hiddenActivationFn, p1.outputActivationFn, p1.lossFunction, p1.optimiser, p1.totalAttributes,numberofHiddenLayersPerChild,neuronesPerLayerChild)
    c2 = Net(p2.epoch, dropoutofChild, p2.hiddenActivationFn, p2.outputActivationFn, p2.lossFunction, p2.optimiser, p2.totalAttributes, numberofHiddenLayersPerChild,neuronesPerLayerChild)
    offspring.append(c1)
    offspring.append(c2)
    n.extend(offspring)
    print("Child1: "+"Epoch: "+ str(c1.epoch))
    print("Child2: "+ str(c2.init_params()))
    print ('---------------------------------------------------CROSSOVER----------------------------------------------------')

    return n

def mutate(n):
    p1 = choice(n)
    drp = 0
    print ('---------------------------------------------------MUTATION----------------------------------------------------')
    #print("Drop out avant la mutation:"  + str(p1.parentDropOut))
    print("Drop out avant la mutation:"  + str(p1.dropout))
    print('------------------------------------------------------------------')
    dropoutPreMutation = p1.dropout

    dropoutPostMutation = 1 - dropoutPreMutation
    print("Drop out après la mutation:"  + str(dropoutPostMutation))
    print('------------------------------------------------------------------')


    #!!!!!!!!! TO DO !!!!!!!!!
    # To loop throught the Neural Network and get the different parameters
    #Select a random hidden layer and updates its drop out value with   value of  dropoutPostMutation = 1 - dropoutPreMutation
    """for i in n:
        if uniform(0, 1) <= 0.1:
            i.epoch += randint(0, 5)
            #i.u1 += randint(0, 5)
   """
    for i in n:
      dropoutPreMutation = i.dropout
      dropoutPostMutation =  1 - dropoutPreMutation
      print('check'+str(dropoutPostMutation))
      i.dropout = dropoutPostMutation
    print ('---------------------------------------------------MUTATION----------------------------------------------------')
    return n




if __name__ == "__main__":
  #lecture et chargement du dataset
  xtrain, xtest, ytrain, ytest, I_sh = kddnsl(show_examples=True)

  #dÃ©finition des variables 
  pmax    = 5 # nombre de couche cachÃ©es
  n       = 41 # taille du vecteur d'entrÃ©e: nombre d'attribut du dataset NSL-KDD

  totalHiddenLayers = randint(1,pmax) #gÃ©nÃ©ration du nombre de couche cachÃ©e au hasard
  neuronesPerLayer  =  randint(n/2,n) #nombre de neurones par couche: selection au hasard pour que le nombre se situe entre n/2 et n ou n est le nombre d'attribut maximum du dataset

  print('Total hidden layer: ' + str(totalHiddenLayers))
  print('------------------------------------------------------------------')
  print('Number of Neurones per layer: ' + str(neuronesPerLayer))
  print('------------------------------------------------------------------')

  population  = 10  # Population
  generation  = 55 # Generation
  batchSize   = 128  # Batch size
  classNumber = 5  # Class number
  threshold   = 0.90  # Threshold

  accuracy_list = []

  #creation du rÃ©seaut de neurones en prenant en consideration la taille de la population
  #dropout value is overwritten during the execution of 
  N = init_net( epoch = 5,dropout =  random.uniform(0,1), hiddenActivationFn = 'relu',outputActivationFn = 'softmax', 
               lossFunction='categorical_crossentropy',optimiser='adadelta' ,totalAttributes = n,totalHiddenLayers = totalHiddenLayers,neuronesPerLayer= neuronesPerLayer,p=population)

  for g in range(generation):
    print('Generation {}'.format(g + 1))
    print('------------------------------------------------------------------')
    print('------------------------------------------------------------------')

    N = fitness(n=N,
                n_c=classNumber,
                i_shape=I_sh,
                x=xtrain,
                y=ytrain,
                b=batchSize,
                x_test=xtest,
                y_test=ytest,
                pMax=pmax
                )
 

   # print(type(N))
  #  print(N)
    accuracies = np.empty(shape=(g))
    N = selection(n=N)
    N = crossover(n=N,pMax = pmax,maxNeurone = n)
    N = mutate(n=N)


    for q in N:
        acc = round((q.accuracy * 100),2)
        accuracy_list.append(acc)
    #    print(accuracy_list)
        if q.accuracy >threshold:
            accuracies[g] = N[0]
            print('Threshold satisfied')
            print(q.init_params())
            print('Best accuracy: {}%'.format(acc))
            exit(code=0)
    #print(accuracy_list)

    print("The best accuracy so far {}%".format(max(accuracy_list)))
    matplotlib.pyplot.plot(np.array(accuracy_list), linewidth=5, color="black")
    matplotlib.pyplot.xlabel("Iteration", fontsize=20)
    matplotlib.pyplot.ylabel("Fitness", fontsize=20)
    matplotlib.pyplot.xticks(np.arange(0, g+1,10), fontsize=15)
    matplotlib.pyplot.yticks(np.arange(0, 101, 10), fontsize=15)
    fig = plt.figure(dpi=100, figsize=(14, 7))
    plt.show()
    plt.savefig('test3.png')  
 