import struct
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame
import math
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import mixture
from sklearn.model_selection import train_test_split
import numpy.linalg as LA
from array import array

#Specify file location Change this
excelfile = 'pathtothisfile/Final_master_without_zip.xlsx'

#Function Definitions
#Read Excel sheet
def readExcelSheet1(excelfile):
    from pandas import read_excel
    return (read_excel(excelfile)).values

#Function to grab first sheet and put into Values list
def readExcelRange(excelfile,sheetname="Sheet1",startrow=1,endrow=1,startcol=1,endcol=1):
    from pandas import read_excel
    values=(read_excel(excelfile, sheetname,header=None)).values;
    return values[startrow-1:endrow,startcol-1:endcol]

#Function to put into Data
def readExcel(excelfile,**args):
    if args:
        data=readExcelRange(excelfile,**args)
    else:
        data=readExcelSheet1(excelfile)
    if data.shape==(1,1):
        return data[0,0]
    elif (data.shape)[0]==1:
        return data[0]
    else:
        return data
    
def writeExcelData(x,excelfile,sheetname,startrow,startcol):
    from pandas import DataFrame, ExcelWriter
    from openpyxl import load_workbook
    df=DataFrame(x)
    book = load_workbook(excelfile)
    writer = ExcelWriter(excelfile, engine='openpyxl') 
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(writer, sheet_name=sheetname,startrow=startrow-1, startcol=startcol-1, header=False, index=False)
    writer.save()
    writer.close()

#Gets sheet names
def getSheetNames(excelfile):
    from pandas import ExcelFile
    return (ExcelFile(excelfile)).sheet_names



def BayesTheorem(queries, covFemales, covMales, meanFemales, meanMales, xnumFemales, xnumMales):
    NF = xnumFemales * (1 / ((2 * np.pi) * np.power(np.linalg.det(covFemales),0.5)) * np.exp(-0.5 * np.dot(np.dot((queries - meanFemales), np.linalg.inv(covFemales)), (queries - meanFemales).T)))
    NM = xnumMales * (1 / ((2 * np.pi) * np.power(np.linalg.det(covMales),0.5)) * np.exp(-0.5 * np.dot(np.dot((queries - meanMales), np.linalg.inv(covMales)), (queries - meanMales).T)))
    PF = NF / (NF + NM)
    PM = NM / (NF + NM)
    gender=''
    PO=0
    if PF > PM :
        gender = positiveClass
        PO=PF
    else :
        gender = negativeClass
        PO=PM
    return gender, PO



def getBayesTheoremQueries(X,covFemales, covMales, meanFemales, meanMales, xnumFemales, xnumMales):
    return np.array(map(lambda x: BayesTheorem(x, covFemales, covMales, meanFemales, meanMales, xnumFemales, xnumMales), X))


#Find indices for selected digits

#Ttest = np.zeros((len(Dtest), 1), dtype=np.uint8)
#Ttrain = np.zeros((len(Dtrain), 1), dtype=np.uint8)
#Puts data into data list. 

#ddata=pandas.read_excel(excelfile)
data=readExcel(excelfile)
print(data)
data.shape
#ddata.head()


#Split between Train and Test
print(data.shape[0]*.70)
Dtrain, Dtest = train_test_split(data, test_size=0.3)
Dtrain.shape
Dtest.shape


# Import dating data - Training
Xtrain=(np.array(Dtrain[:,:67].astype('int32')))   #Features
Ttrain=np.array(Dtrain[:,67].astype('int32'))   #Match

# Import dating data - Testing
Xtest=(np.array(Dtest[:,:67].astype('int32')))   #Features
Ttest=np.array(Dtest[:,67].astype('int32'))   #Match


#Run PCA on the training data
#Z part of XZCVPR
meanZ = np.mean(Xtrain,axis=0);
print(len(meanZ))
print(meanZ)
meanZ.shape


print(np.amin(meanZ))
print(np.amax(meanZ))

Z=Xtrain-meanZ;
print(Z)
Z.shape #check dimensions of Z
print(np.amin(Z))   #check min and max of Z
print(np.amax(Z))


np.mean(Z,axis=0)

#plotting out the mean vector against the dimensions. 
plt.plot(meanZ)
plt.show()
#write meanVector to Excel



#C part of XZCPVR
C=np.cov(Z,rowvar=False);
print(C)

C.shape  #Checking that dimensions are 784x784
CT=C.T
print(CT)


#V part of XZCVPR
[E,V]=LA.eigh(C);
print(E,'\n\n',V)

row=V[-1,:];col=V[:,-1];
np.dot(C,row)-(E[-1]*row)
np.dot(C,col)-(E[-1]*col)

E=np.flipud(E);
V=np.flipud(V.T);   #Does this put it in Descending order?
row=V[0,:]; #Check once again
np.dot(C,row)-(E[0]*row)

 #P part of XZCVPR
P=np.dot(Z,V.T);
print(P) #Principal components
P.shape    #checking the dimension of P
np.mean(P, axis=0) #should be close to 0. Which it is...

#Bayesian 2D


positiveClass=1
negativeClass=0
P2D = P[:,0:2]
V2D = V[0:2,:]
P2D.shape
Ttrain.shape
positives=P2D[Ttrain[:] == positiveClass]
negatives=P2D[Ttrain[:] == negativeClass]
Np=positives.shape[0]
Nn=negatives.shape[0]


mup=np.mean(positives,axis=0)
mun=np.mean(negatives,axis=0)

Zp=positives-mup
Zn=negatives-mun

cp=np.cov(Zp,rowvar=False)
cn=np.cov(Zn,rowvar=False)
print 'P2D shape: ', P2D.shape

bayesList=getBayesTheoremQueries(P2D,cp,cn,mup,mun,Np,Nn)


bayesresult=bayesList[:,0]
print 'This is bayesresult: '
print bayesresult


predictions_list = bayesresult.tolist()
actuals_list = Ttest.tolist()

correct_predictions = 0

for i in range(len(actuals_list)):
    if float(predictions_list[i]) == float(actuals_list[i]):
        correct_predictions = correct_predictions + 1

print correct_predictions
accurarcy = float(correct_predictions)/2514

print 'Number of correct predictions: ', correct_predictions
print 'Number of total vectors: ', len(actuals_list)
print accurarcy

xp=Xtest[2,:]
tp=Ttest[2]
print(Ttest[2])
xp.shape
zp=xp-meanZ
pp=np.dot(zp,V2D.T)
rp=np.dot(pp,V2D)
xrecp=rp+meanZ
print(xrecp)
print(pp)


#xnt, tnt = load_mnist(dataset="testing",selecteddigits=[negativeClass])
xn=Xtest[0,:]
zn=xn-meanZ
pn=np.dot(zn,V2D.T)
rn=np.dot(pn,V2D)
xrecn=rn+meanZ
print(xrecn)
print(pn)

[resultlabelBp,resultprobBp]=BayesTheorem(np.array([pp.T]),cp,cn,mup,mun,Np,Nn)
[resultlabelBn,resultprobBn]=BayesTheorem(np.array([pn.T]),cp,cn,mup,mun,Np,Nn)

print 'resultlabelBp: ', resultlabelBp
print 'resultprobBp: ', resultprobBp
print 'resultlabelBn: ', resultlabelBn
print 'resultprobBn: ', resultprobBn
#5 variables scatter
np.random.seed(0)
randomorder=np.random.permutation(np.arange(len(Ttrain)))
opacity=0.25
cols = np.zeros((len(Ttrain), 4))
cols[Ttrain==0]=[1,0,0,opacity]
cols[Ttrain==1]=[0,1,0,opacity]
fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,facecolor='black')
ax.scatter(P[randomorder,1], P[randomorder,0], s=50,linewidths=0,facecolors=cols[randomorder,:])
ax.set_aspect('equal')
plt.gca().invert_yaxis()
plt.show()

colors = np.where(Ttrain==positiveClass,'r','b')
print 'Colors shape: ', colors.shape
plt.scatter(P2D[:,0], P2D[:,1],color=colors[:])
plt.show()


