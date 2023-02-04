import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
x

Train = CS[:5000,:]
Test  = CS[5000:,:]
AnsTrain = ANS[:5000]
AnsTest  = ANS[5000:]

NeuralNet = MLPClassifier(solver='lbfgs', alpha=1e0, hidden_layer_sizes=(5))
NeuralNet.fit(Train.tolist(), (AnsTrain >= 1).tolist())

# =====================================

Probs = np.array(NeuralNet.predict_proba(Test.tolist()))[:,1]
Guess = np.array(NeuralNet.predict(Test.tolist()))

Ans   = (AnsTest >= 1).tolist()
nAns  = (AnsTest == 0).tolist()

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

print('=============================================')
# print(f'Fraction existing = {np.sum(Ans)/len(Ans)}')
print(f'Trivial Score = {max(1.0-np.sum(Ans)/len(Ans),np.sum(Ans)/len(Ans))}')
print(f'Score = {(np.sum(Probs[Ans] >= 0.5) + np.sum(Probs[nAns] < 0.5))/len(Ans)}')
# print(f'Score = {(np.sum(Probs[Ans]) + np.sum(1.0-Probs[[~r for r in Ans]]))/N}')
print('=============================================')
print('Confusion Matrix:')
print(confusion_matrix(Ans, Guess))

ProbCorrPlot(Probs, Ans)

N, n = Test.shape

plt.figure()
plt.plot(Test.reshape(-1,), '.k')
plt.vlines(range(0,N*n,n), -0.4, 0.2)

# plt.hlines([-0.4, -0.41], 0, N*n, colors=['r','r'], linewidth=3)
# [plt.hlines(-0.4, n*idx, n*(idx+1), colors='b', linewidth=3) for idx in range(N) if bool(Guess[idx])]
# [plt.hlines(-0.41, n*idx, n*(idx+1), colors='g', linewidth=3) for idx in range(N) if Ans[idx]]

plt.xlabel('Index', fontsize=16)
plt.ylabel('Transmission', fontsize=16)
plt.tight_layout()
plt.show()

# plt.figure()
# plt.plot(TrueDataTest.reshape(-1), '-b')
# plt.plot(RawDataTest.reshape(-1), '.k')
# plt.vlines(NP*np.arange(N), 0, 3)
# plt.title('Cross-section',fontsize=18)
# plt.xlabel('Index',fontsize=15)
# plt.ylabel('Cross-section',fontsize=15)
# plt.show()
