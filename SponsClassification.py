# coding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import numpy as np

import glob
from sklearn.externals import joblib

from sklearn.linear_model import SGDClassifier

prediction_dict = {1:'Sponsored' , 0:'Non Sponsored'}
def read_file(filepath):
	fd = open(filepath,'r')
	content = fd.read()
	fd.close()
	return content

def read_data(data_dir,split):
	data = []
	label = []

	for f in glob.glob(data_dir + "*.txt"):
	    #print (f)
	    label_tag = f.split('/')[-1:][0]
	    label_name = label_tag.split('_')[:-1][0]
	    if label_name == 'sponsored':
	    	#print(label_name + ': 1 \n')
	    	label.append(1)
	    else:
	    	#print(label_name + ': 0 \n')
	    	label.append(0)

	    fd = open(f, 'r', encoding='utf-8')
	    blog = fd.read()

	    data.append(blog)
	    fd.close()
	train_len = int(len(data)*split)

	Xtrain = data[0: train_len]
	ytrain = label[0: train_len]
	Xtest = data[train_len : len(data)]
	ytest = label[train_len : len(data)]

	return (Xtrain,ytrain,Xtest,ytest)


def save_data(Xtrain,ytrain,Xtest,ytest,pkl_dataname):
	joblib.dump((Xtrain,ytrain,Xtest,ytest),pkl_dataname)

def load_data(pkl_dataname):
	(Xtrain,ytrain,Xtest,ytest) = joblib.load(pkl_dataname)
	return (Xtrain,ytrain,Xtest,ytest)

def NB_train_model(Xtrain,ytrain,pkl_modelname):
	text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
	text_clf = text_clf.fit(Xtrain, ytrain)
	joblib.dump(text_clf, pkl_modelname)

def SVM_train_model(Xtrain,ytrain,pkl_modelname):
	text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=200, random_state=42))])

	text_clf_svm = text_clf_svm.fit(Xtrain, ytrain)
	joblib.dump(text_clf_svm, pkl_modelname)



def load_model(pkl_modelname):
	model = joblib.load(pkl_modelname)
	return model

def validate_model(Xtest,ytest,model):
	predicted = model.predict(Xtest)
	validation_acc = np.mean(predicted == ytest)

	return (predicted,validation_acc)

def content_classifier(text_clf,model):
	prediction = model.predict(text_clf)
	return prediction


def train_main():
	
	split = 0.7
	data_dir = 'data/'
	pkl_filename = 'CIO_spons_data.pkl'
	NB_pkl_modelname = 'CIO_NB_Classifier.pkl'
	SVM_pkl_modelname = 'CIO_SVM_Classifier.pkl'

	# Read data from desired directory
	(Xtrain_1,ytrain_1,Xtest_1,ytest_1) = read_data('ComputerWorld/CW_more_unspons/',0.1)
	print (len(Xtest_1))



	# Read data from desired directory
	(Xtrain,ytrain,Xtest,ytest) = read_data(data_dir,split)
	print (len(Xtrain))

	# Save data for later use
	save_data(Xtrain,ytrain,Xtest,ytest,pkl_filename)
	
	# Load dataset has been saved for ML models in pickle format
	(Xtrain,ytrain,Xtest,ytest) = load_data(pkl_filename)

	# Train Naive Bayes Model
	NB_train_model(Xtrain,ytrain,NB_pkl_modelname)
	SVM_train_model(Xtrain,ytrain,SVM_pkl_modelname)

	# Load Trained Model
	NB_model = load_model(NB_pkl_modelname)
	SVM_model = load_model(SVM_pkl_modelname)

	# Validate Model
	NB_predicted , NB_acc = validate_model(Xtest,ytest,NB_model)
	SVM_predcited, SVM_acc = validate_model(Xtest,ytest,SVM_model)

	print ('NB Accuracy: ' + str(NB_acc))
	print ('SVM Accuracy: ' + str(SVM_acc))

	print (confusion_matrix(ytest,NB_predicted))
	print (confusion_matrix(ytest,SVM_predcited))

	NB_predicted_1 , NB_acc_1 = validate_model(Xtest_1,ytest_1,NB_model)
	SVM_predcited_1, SVM_acc_1 = validate_model(Xtest_1,ytest_1,SVM_model)

	print ('NB Accuracy: ' + str(NB_acc_1))
	print ('SVM Accuracy: ' + str(SVM_acc_1))

	print (confusion_matrix(ytest_1,NB_predicted_1))
	print (confusion_matrix(ytest_1,SVM_predcited_1))

	# Classifiy a blog

def classification_main():

	NB_pkl_modelname = 'CIO_NB_Classifier.pkl'
	SVM_pkl_modelname = 'CIO_SVM_Classifier.pkl'
	# 365
	query = '/home/msaleem/Ahmer/Sponsored_Project/sklearn-classification/test/unsponsored_460.txt'

	# Load Trained Model
	NB_model = load_model(NB_pkl_modelname)
	SVM_model = load_model(SVM_pkl_modelname)


	blog = read_file(query)
	NB_prediction = content_classifier([blog],NB_model)[0]
	SVM_prediction = content_classifier([blog],SVM_model)[0]

	print ('NB Prediction : ' + prediction_dict[NB_prediction])
	print ('SVM Prediction : ' + prediction_dict[SVM_prediction])

if __name__ == "__main__":
    train_main()
    #classification_main()
