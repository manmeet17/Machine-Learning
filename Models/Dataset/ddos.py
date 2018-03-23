import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

df=pd.read_csv('kddcup.csv')

columns=['duration',
'protocol_type',
'service',
'flag',
'src_bytes',
'dst_bytes',
'land',
'wrong_fragment',
'urgent',
'hot',
'num_failed_logins',
'logged_in', 
'num_compromised',
'root_shell',
'su_attempted',
'num_root',
'num_file_creations',
'num_shells',
'num_access_files',
'num_outbound_cmds',
'is_host_login',
'is_guest_login', 
'count',
'srv_count',
'serror_rate',
'srv_serror_rate',
'rerror_rate',
'srv_rerror_rate',
'same_srv_rate',
'diff_srv_rate',
'srv_diff_host_rate',
'dst_host_count',
'dst_host_srv_count',
'dst_host_same_srv_rate',
'dst_host_diff_srv_rate',
'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate',
'dst_host_serror_rate',
'dst_host_srv_serror_rate',
'dst_host_rerror_rate',
'dst_host_srv_rerror_rate',
'DDOS'
]

df=pd.read_csv('kddcup.csv',names=columns)
df['DDOS']=df['DDOS'].map(lambda x: x.rstrip('.'))
# print (df['DDOS'].value_counts())

mapped={
"smurf": 1,
'neptune':2,
'normal':3,
'back': 4,
'satan':5,
'ipsweep':6,
'portsweep':7,
'warezclient':8,
'teardrop':9,
'pod':10,
'nmap' :11,
'guess_passwd':12,
'buffer_overflow':13,
'land':14,
'warezmaster':15,
'imap': 16,
'rootkit' :17,
'loadmodule':18,
'ftp_write':19,
'multihop': 20,
'phf': 21,
'perl': 22,
'spy': 23
}

X=df.drop(['DDOS','protocol_type','service','flag'],axis=1)
y=df['DDOS']
y=y.map(mapped)
y=y.astype('float')
print (X.info())
svm=SVC(C=1.0,kernel='rbf')
naive=GaussianNB()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print (svm.score(X_test,y_test))