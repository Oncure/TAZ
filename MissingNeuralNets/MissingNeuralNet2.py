import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

N = 20000 # Number of samplings

NP  = 50 # Number of points per sample
DE = 15
dE = DE/(NP+1)

SF = 8

S  = 15
BG = 15

Ma = 1
Mb = 0.1
MLS = 20

E = np.linspace(dE, DE-dE, NP)

def Noise(X):   return (np.random.poisson(S*X+BG)-BG)/S

NumRes = np.random.poisson(DE/MLS, size=N)
TrueDataTrain = np.zeros((N,NP))
for n in range(N):
    # El = np.random.uniform(low=0.0, high=DE, size=NumRes[n]).reshape(1,-1)
    El = np.random.normal(DE/2, DE/SF, size=NumRes[n]).reshape(1,-1)
    Ga = Ma * abs(np.random.normal(size=NumRes[n])).reshape(1,-1)
    Gb = Mb * abs(np.random.normal(size=NumRes[n])).reshape(1,-1)
    TrueDataTrain[n,:] = np.sum(Ga*Gb / ((E.reshape(-1,1)-El)**2 + ((Ga+Gb)/2)**2), axis=1)
RawDataTrain = Noise(TrueDataTrain)
# RawDataTrain = TrueDataTrain

NeuralNet = MLPClassifier(solver='lbfgs', alpha=1e0, hidden_layer_sizes=(10,3))
NeuralNet.fit(RawDataTrain.tolist(), (NumRes >= 1).tolist())

# =====================================

NumRes = np.random.poisson(DE/MLS, size=N)
TrueDataTest = np.zeros((N,NP))
for n in range(N):
    # El = np.random.uniform(low=0.0, high=DE, size=NumRes[n]).reshape(1,-1)
    El = np.random.normal(DE/2, DE/SF, size=NumRes[n]).reshape(1,-1)
    Ga = Ma * abs(np.random.normal(size=NumRes[n])).reshape(1,-1)
    Gb = Mb * abs(np.random.normal(size=NumRes[n])).reshape(1,-1)
    TrueDataTest[n,:] = np.sum(Ga*Gb / ((E.reshape(-1,1)-El)**2 + ((Ga+Gb)/2)**2), axis=1)
RawDataTest = Noise(TrueDataTest)
# RawDataTest = TrueDataTest

Probs = np.array(NeuralNet.predict_proba(RawDataTest.tolist()))[:,1]
Ans   = (NumRes >= 1).tolist()
nAns  = (NumRes == 0).tolist()

def ProbCorrPlot(Prob_Guess_All, Answer):
    nBin = round(np.sqrt(len(Answer)))
    edges = np.linspace(0.0, 1.0, nBin+1)
    X = (edges[:-1] + edges[1:])/2
    Prob_Guess_Type = Prob_Guess_All
    Prob_Guess_Cor  = Prob_Guess_Type[Answer]
    Count_All = np.histogram(Prob_Guess_Type, bins=edges)[0]
    Count_Cor = np.histogram(Prob_Guess_Cor , bins=edges)[0]

    X2         = X[Count_All != 0]
    Count_All2 = Count_All[Count_All != 0]
    Count_Cor2 = Count_Cor[Count_All != 0]
    Prob_Ans_Cor = Count_Cor2/Count_All2
    Prob_Ans_Cor_SD = 1.0 / np.sqrt(Count_All2)

    # Plotting:
    plt.figure()
    plt.errorbar(X2,Prob_Ans_Cor,Prob_Ans_Cor_SD,capsize=3,ls='none',c='k')
    plt.scatter(X2,Prob_Ans_Cor,marker='.',s=14,c='k')
    plt.plot([0,1],[0,1],':b',lw=2,label='Ideal Correlation')
    Norm = 0.25/np.max(Count_All)
    plt.fill_between(X,Norm*Count_All,alpha=0.8,color=(0.1,0.5,0.1),label='Probability Density')
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title('Probability Accuracy',fontsize=18)
    plt.xlabel('Predicted Probability',fontsize=15)
    plt.ylabel('Fraction of Correct Assignments',fontsize=15)
    plt.legend(fontsize=10)

    plt.show()

print(Probs)

# print(f'Fraction existing = {np.sum(Ans)/len(Ans)}')
print(f'Trivial Score = {max(1.0-np.sum(Ans)/len(Ans),np.sum(Ans)/len(Ans))}')
print(f'Score = {(np.sum(Probs[Ans] >= 0.5) + np.sum(Probs[nAns] < 0.5))/N}')
# print(f'Score = {(np.sum(Probs[Ans]) + np.sum(1.0-Probs[[~r for r in Ans]]))/N}')

ProbCorrPlot(Probs, Ans)

plt.figure()
plt.plot(TrueDataTest.reshape(-1), '-b')
plt.plot(RawDataTest.reshape(-1), '.k')
plt.vlines(NP*np.arange(N), 0, 3)
plt.title('Cross-section',fontsize=18)
plt.xlabel('Index',fontsize=15)
plt.ylabel('Cross-section',fontsize=15)
plt.show()
