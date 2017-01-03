from sklearn import svm,preprocessing
from sklearn.cross_validation import cross_val_score
import numpy as np
import sys
import collections


y_list = []
X_list = []

print("Reading Files")
'''
with open('train_label.csv') as f:
	for line in f:
		y_list.append(int(line))

with open('gist/train_gists.csv') as f:
	for line in f:
		X_list.append(eval(line.strip()))

print("Converting to Arrays")
X = np.array(X_list)
y = np.array(y_list)

np.savez('gist.npz',X=X,y=y)
'''
gists = np.load('gist.npz')

X = gists['X']
#print X.mean(axis=0)
#print X.std(axis=0)
X = preprocessing.scale(X)
#print X.mean(axis=0)
#print X.std(axis=0)
y = gists['y']

cutoff = int(len(X) * 0.25)

X_val = X[:cutoff,:]
y_val = y[:cutoff]
#print y_list

X_train = X[cutoff:,:]
y_train = y[cutoff:]

cnt = collections.Counter()
for y_i in y:
	cnt[y_i] += 1./len(y)

print cnt
#print (X_train,y_train)
#X = [[0, 0], [1, 1]]
#y = [0, 1]

print("Training svm")
#
class_weight={1:0.45,2:0.5,5:0.6,4:1.5,6:4,8:3,7:8}

clf = svm.SVC(kernel='rbf',degree=2,shrinking=False,C=0.8,gamma='auto',class_weight=class_weight)

#print cross_val_score(clf, X, y, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#sys.exit()

print clf.fit(X_train, y_train)
predict_val = clf.predict(X_val)
#predict_val = np.load('rbf_deg_3.npz')['predict_val']
#np.savez('garbage.npz', clf=clf, predict_val=predict_val)
print predict_val

cnt = collections.Counter()
for y_i in predict_val:
	cnt[y_i] += 1./len(predict_val)

print cnt
print np.sum(predict_val == y_val) * 100. /len(y_val)


#print np.sum(predict_val == 1)