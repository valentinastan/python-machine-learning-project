from cgitb import reset
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#_____________________________________________________________________________________________________________________________________________
# Se va folosi un subset de date LendingClub DataSet obtinut de aici: https://www.kaggle.com/wordsforthewise/lending-club
# LendingClub este o companie americană de creditare, cu sediul în San Francisco, California


# Scopul proiectului:
# Avand în vedere date istorice despre împrumuturile acordate, cu informații despre dacă împrumutatul a rămas sau nu 
# în stare de nerambursare (debitare), voi construi un model care să poată prezice dacă un împrumutat nu își va rambursa 
# împrumutul. Astfel, în viitor, când se obține un nou client potențial, se va putea evalua dacă este sau nu probabil să ramburseze împrumutul.
#______________________________________________________________________________________________________________________________________________

def reset_pyplot():
  plt.figure().clear()
  plt.close()
  plt.cla()
  plt.clf()

data_info = pd.read_csv(r'C:\Users\valen\Desktop\MASTER\MASTER-an2-sem2\TMWA\TMWA-Proiect\lending_club_info.csv', encoding='utf-8', index_col='LoanStatNew')

#print(data_info.loc['revol_util']['Description'])

def feat_info(col_name):
  print(data_info.loc[col_name]['Description'])

# feat_info('mort_acc')

df = pd.read_csv(r'C:\Users\valen\Desktop\MASTER\MASTER-an2-sem2\TMWA\TMWA-Proiect\lending_club_loan_two.csv', encoding='utf-8')
# df.info()

#_____Analiza datelor__________________________________________________________________________________________________________________________________________________________________________________

#countplot
sns.countplot(x='loan_status', data=df)
plt.savefig('1-countplot.png')
reset_pyplot()

#histograma
plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False,bins=40)
plt.xlim(0,45000)

plt.savefig('2-histogram.png')
reset_pyplot()

#correlatie intre toate variabilele
corr = df.corr()
corr.to_csv('Correlation.csv', index=True)

#corelatie heatmap
plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.ylim(10, 0)

plt.savefig('3-corelatie.png')
reset_pyplot()

#installment are corel 0.94 cu loan amount
feat_info('installment')
feat_info('loan_amnt')

#scatterplot
sns.scatterplot(x='installment',y='loan_amnt',data=df)
plt.savefig('4-scatterplot.png')
reset_pyplot()

#boxplot pt a vedea relatia dintre the loan_status si Loan Amount
sns.boxplot(x='loan_status',y='loan_amnt',data=df)
plt.savefig('5-boxplot.png') #impr mai -> nereturnari mai multe
reset_pyplot()

#Statistici descriptive pt loan amount, grupat dupa loan_status.
descr = df.groupby('loan_status')['loan_amnt'].describe()
descr.to_csv('Descriptive_Statistics.csv', index=True)

#coloanele Grade si SubGrade pe care LendingClub le-a atribuit in functie de loans
print(sorted(df['grade'].unique()))
print(sorted(df['sub_grade'].unique()))

#countplot per grade
sns.countplot(x='grade',data=df,hue='loan_status')
plt.savefig('6-countplot-per-grade.png')
reset_pyplot()


#Display count plot per subgrade
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )
plt.savefig('7-countplot-per-sub-grade.png')
reset_pyplot()

#cu hue
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' ,hue='loan_status')
plt.savefig('8-countplot-per-sub-grade-hue.png')
reset_pyplot()


#Pare ca subgradele F si G nu platesc inapoi f des. Le voi izola si recreat countplot-ul doar pt ele
f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]
plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')
plt.savefig('9-countplot-per-sub-grade-F-and-G.png')
reset_pyplot()

#Crearea unei noi coloane 'load_repaid' care contine: 1 daca loan status e "Fully Paid" si 0 daca e "Charged Off"
print(df['loan_status'].unique())
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
print(df[['loan_repaid','loan_status']])

#Crearea unui bar plot pt a vedea corelatia dintre coloane si noua coloana loan_repaid.
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
plt.figure(figsize=(12,4))
plt.savefig('10-barplot-correllation-loar-repaid')
reset_pyplot()

#_____Preprocesarea datelor__________________________________________________________________________________________________________________________________________________________________________________

#Stergere sau adaugare unde sunt date lipsa
#Stergerea datelor nenecesare sau repetitive
#Conversia valorilor string care s categorice in variabile dummy
print('-----------------------------Data PreProcessing-------------------------')

print(df.head())
#length of the df
print('-----Length: ', len(df))

#afisarea nr total de valori lipsa per coloana.
print('-----afisarea nr total de valori lipsa per coloana-----')
print(df.isnull().sum())

#Conversia acestei serii sa fie afisata in raport procentual in dataframe
print('---Conversia acestei serii sa fie afisata in raport procentual in dataframe----')
print(100* df.isnull().sum()/len(df))

#Analiza emp_title si emp_length pt a vedea daca le eliminam din df
print('\n ---Info col emp_title')
feat_info('emp_title')
print('\n ---Info col emp_length')
feat_info('emp_length')

#Calcularea nr de valori unice pt numele jobului angajatului
print('No of unique employment job titles: ', df['emp_title'].nunique())
print(df['emp_title'].value_counts())

#Realistic..sunt f multe titluri unice pentru a incerca convertirea lor in variabile dummy. Se va sterge coloana
df = df.drop('emp_title',axis=1)

#Crearea unui count plot pt coloana emp_length. Valorile se sorteaza.
print(sorted(df['emp_length'].dropna().unique())) #anii de munca

emp_length_order = [ '< 1 year',
  '1 year',
  '2 years',
  '3 years',
  '4 years',
  '5 years',
  '6 years',
  '7 years',
  '8 years',
  '9 years',
  '10+ years'
]

plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order)

plt.savefig('11-barplot-emp-length')
reset_pyplot()

#Crearea unui countplot cu hue pt Fully Paid si Charged Off
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')

plt.savefig('12-barplot-emp-length-with-hue')
reset_pyplot()

# Acest lucru încă nu ne informează cu adevărat dacă există o relație puternică între durata angajării
# și nerambursarea creditului, aasa ca ne va interesa procentul de nerambursare pe categorie.
# O sa aflam ce procent din persoane din fiecare categorie de angajare nu și-au rambursat împrumutul.
emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']

emp_len = emp_co/emp_fp
print("Ratio, won't pay back: ", emp_len)

emp_len.plot(kind='bar')
plt.savefig('13-barplot-won`t pay back ratio')
reset_pyplot()

# Ratele Charge off sunt f similare cu employment lengths (19 si 20%). Voi sterge coloana emp_length.
df = df.drop('emp_length',axis=1)

# Vizualizare DataFrame pt a vedea ce coloane inca au date lipsa
print(df.isnull().sum())

#Analiza coloanelor title si purpose pt a vedea daca exista informatie repetata
print(df['purpose'].head(10))
feat_info('purpose')
print(df['title'].head(10))
feat_info('title')

# Coloana titlu e pur si simplu un string din coloana purpose. Stergere coloana title
df = df.drop('title',axis=1)

# Verificare a ce reprezinta coloana mort_acc
feat_info('mort_acc')

# Create value_counts pt coloana mort_acc
print(df['mort_acc'].value_counts()) #avem date null pe 10%din date

#cautam ce col sunt f mult corelate cu col mort_acc
print("----Correlation with the mort_acc column----")
print(df.corr()['mort_acc'].sort_values())

# Se pare ca col total_acc se coreleaza cu mort_acc
# Voi folosi fillna(). Se va grupa dataframe-ul dupa total_acc 
# si se va calcula valoarea medie pt mort_acc per fiecare intrare total_acc .


print("----Mean of mort_acc column per total_acc----")
print(df.groupby('total_acc').mean()['mort_acc'])

# Se inlocuiesc valorile lipsa mort_acc pe vaza valorii lor aferente din total_acc.
# Daca mort_acc lipseste, valoarea lipsa va fi inlocuita cu
# valoarea medie corespunzatoare din total_acc din seria creata mai sus. 

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
print('total_acc_avg: ', total_acc_avg[2.0])

def fill_mort_acc(total_acc,mort_acc):
  if np.isnan(mort_acc):
    return total_acc_avg[total_acc]
  else:
    return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
print(df.isnull().sum())


# revol_util si the pub_rec_bankruptcies au date lipsa, acestea reprezentand mai putin de
# 0.5% din toata seria de date. Se vor sterge liniile care au valori lipsa in aceste coloane

df = df.dropna()
print(df.isnull().sum())

#_____Variabile Categorice si Dummy Variables__________________________________________________________________________________________________________________________________________________________________________________

print('\n \n _________________Variabile Categorice si Dummy Variables_________________________')
# ne ocupam de valorile string din cauza coloanelor care au categorii.

print('\n', df.select_dtypes(['object']).columns)

# Conversia coloanei term in val numerice intregi 36 sau 60
feat_info('term')
print('\n term: ', df['term'].value_counts())

df['term'] = df['term'].apply(lambda term: int(term[:3]))
print(df['term'])

# coloana grade 
# Se stie ca grade este parte a coloanei sub_grade, asa ca voi sterge coloana grade
df = df.drop('grade',axis=1)

# Conversia coloanei subgrade in variabile dummy. Apoi concatenarea lor in coloane noi in df-ul original
# Coloana originala subgrade va fi stearsa.

subgrade_dummies = pd.get_dummies(df['sub_grade'], drop_first=True)

df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
print('\n Columns: ', df.columns)

#  verification_status, application_type,initial_list_status,purpose 
# Conversia acestor coloane: ['verification_status', 'application_type','initial_list_status','purpose'] 
# in variabile dummy si concatenarea lor cu df-ul original
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)
print('\n Columns: ', df.columns)

# home_ownership
# Verificarea value_counts pt coloana home_ownership.
df['home_ownership'].value_counts()

# Conversie in variabile dummy, dar inlocuirea NONE si ANY cu OTHER, ca la final sa avem doar 4 categorii: MORTGAGE, RENT, OWN, OTHER.
# concatenarea lor cu df-ul original.

df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

# address
# Crearea unei coloane zip_code din coloana address din data set.
# Se va extrahe codul postal din coloana adresa
df['zip_code'] = df['address'].apply(lambda address:address[-5:])

dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)

# issue_d 
# Voi sterge aceasta coloana, nefiind utila pt analiza curenta (nu ne trebuie data aceasta)
feat_info('issue_d')
df = df.drop('issue_d',axis=1)

# earliest_cr_line
# Se va extrage anul din această coloana folosind o funcție .apply, apoi va fi convertit la tip numeric
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
print(df['earliest_cr_year'])
df = df.drop('earliest_cr_line',axis=1)

print(df.select_dtypes(['object']).columns)


#____Train Test Split__________________________________________________________________________________________________________________________________________________________________________________

print('\n \n _____________________ Train Test Split_________________________')

# Import train_test_split din sklearn
from sklearn.model_selection import train_test_split

# stergerea coloanei load_status creata anterior,pt ca e duplicata cu coloana loan_repaid.
# Voi folosi col loan_repaid pentru ca are valori intre 0 si 1
df = df.drop('loan_status',axis=1)

# Setarea variabilelor X si y pt .values of the features and label.
X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values

print(len(df))

# Impartirea intre train/test cu test_size=0.2 si random_state = 101
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

# Normalizare
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Crearea Modelului
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation,Dropout
from tensorflow.python.keras.constraints import max_norm

# Crearea unui model secvential care va fi antrenat de date.
# Modelul va avea (nr coloane=)78 --> 39 --> 19--> 1 output neuron.

model = Sequential()

# input layer
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='sigmoid')) #pt a avea val 0-1

# Compilare model
model.compile(loss='binary_crossentropy', optimizer='adam')#binary clasification

print(X_train.shape)

# Modelul va antrena datele pt 25 epochs. 
# Adaugare datele in validare pt reprezentare mai usoara. Adaugare batch_size = 256.

model.fit(x=X_train, 
  y=y_train, 
  epochs=25,
  batch_size=256,
  validation_data=(X_test, y_test), 
) 

#____Evaluarea performantei modelului__________________________________________________________________________________________________________________________________________________________________________________

# ---------------Evaluarea performantei modelului--------------------
#  Reprezentare validation loss versus training loss
losses = pd.DataFrame(model.history.history) #history of losses
losses[['loss','val_loss']].plot() #trains vs validation loss
plt.savefig('13-evaluation')
reset_pyplot()

# Crearea de predicții din setul X_test și afișarea unui raport de clasificare și o matrice de confuzie pentru setul X_test
from sklearn.metrics import classification_report,confusion_matrix

predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions)) 
#accuracy 0.89
#f1-score = 0.61

confusion_matrix(y_test,predictions)


# Avand acest client, i se va oferi acestuia creditul?
import random
random.seed(101)
random_ind = random.randint(0,len(df)) #index random

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
print('New customer: ', new_customer)

print('\n PREDICT', model.predict_classes(new_customer.values.reshape(1,78)))

# A ajuns aceasta persoana sa isi ramburseze creditul?
print(df.iloc[random_ind]['loan_repaid'])
