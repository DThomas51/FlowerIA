import numpy as np 
 
x_entrer = np.array (([3,1.5],[2,1],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4,1.5]),dtype=float) 
y = np.array (([1],[0],[1],[0],[1],[0]),dtype=float) # 1 = Rouge / 0 = Bleu

x_entrer = x_entrer/np.amax(x_entrer,axis=0)

X = np.split(x_entrer,[8])[0]
xPrediction = np.split(x_entrer,[8])[1]

class Neural_Network(object):
    def _init_(self):
        self.imputSize = 2 # Neurone
        self.outputSize = 1
        self.hiddenSize = 3

        self.W1 = np.random.randn(self.imputSize, self.hiddenSize) # Matrice 2 par 3 ( Valeur des Synapse Aléatoire)
        self.W2 = np.random.randn(self.hiddenSize,self.outputSize) # Matrice 3 par 1 ( Valeur des Synapse Aléatoire)
    
    def forward(self,X):
        
        self.z = np.dot(X,self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.W2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self,s):
        return 1/(1+np.exp( -s ))

NN = Neural_Network()

o = NN.forward(X)

print("Sortie prédict de l'IA : \n" + str(o))
print("vrai sortie : \n" + str(o))
 


