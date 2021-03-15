import numpy as np 
 
x_entrer = np.array (([3,1.5],[2,1],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4,1.5]),dtype=float) 
y = np.array (([1],[0],[1],[0],[1],[0],[1],[0]),dtype=float) # 1 = Rouge / 0 = Bleu

x_entrer = x_entrer/np.amax(x_entrer,axis=0)

X = np.split(x_entrer,[8])[0]
xPrediction = np.split(x_entrer,[8])[1]

class Neural_Network(object):
    def __init__(self):
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

    def sigmoidPrime(self,s):
        
        return s * (1-s)

    def backward(self,X,y,o):

        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)

        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        self.W1 += X.T.dot(self.z2_delta) # Mise à jour des poids de W1
        self.W2 += self.z2.T.dot(self.o_delta) # Mise à jour des poids de W2

    def train(self,X,y): # Fonction pour entrainer l'ia a mettre à jour le poid des Synapse
        o = self.forward(X)
        self.backward(X,y,o) # X = valeur d'entée y = valeur qu'on attend o = output venant de forward
    
    def predict(self):
        print("Donnée apres l'entrainement de l'ia : ")
        print("Entrée : \n" + str(xPrediction)) 
        print("Sortie : \n" + str(self.forward(xPrediction)))


        if(self.forward(xPrediction) < 0.5 ):
           print("La fleur est bleu !! \n")
        else:
           print("La fleur est rouge !! \n")    


NN = Neural_Network()
 
for i in range(30000): # Utiliser la Boucle pour permettre à l'ia d'apprendre pour ce rapprocher de la vrai valeur
    print( "#" + str(i) + "\n" ) 
    print("Valeurs d'entrée : \n" + str(X))
    print("Sortie Actuelle : \n " + str(y))
    print("Sortie Prédite par l'ia : \n " + str(np.matrix.round(NN.forward(X),2))) # np.matrix... arrondie a deux décimal
    print("\n")
    NN.train(X,y)
NN.predict()



# Rétropropagation (Utilisation d'une fonction objective pour modifier le poid des Synapse pour rendre le résultat le plus proche possible de sa valeur Réel)
# Utilisation de l'Algoritme du gradient pour savoir comme le poid des Synapse joue sur la valeur d'entrée.
# -Calculer la marge d'erreur (imput-output) = erreur 
# -Appliquer la dérivée de la sigmoid à l'erreur =erreur delta
# -Multiplication matricielle en W2 et l'erreur delta = erreur 2 cachée
# -Appliquer la derivée de la sigmoid à l'erreur 2 = erreur 2 delta cachée
# -Ajuster W1 et W2

