import sys
import numpy as np
import copy
import pickle
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from Dataset_loader import MNIST
import pandas as pd
import networkx as nx
from Functions import create_graph, create_graphs,Normalized_Cuts_all_graphs,Normalized_Cuts,create_pixals_counter2,normalized_factor,bp,bp2,normalized_predict,run_test,run_test2,print_matrix,image_binary
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
data = MNIST('.')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)
X = train_img
y = train_labels
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.1)
X_train=X_test
y_train=y_test
limit = 10
number_of_loops=1
upper_bound=0.1
low_bound=0.1
number_of_images=1
number_of_edges=25

old_stdout = sys.stdout
results_file = open("results.log","w")
sys.stdout = results_file


list_of_graphs=create_graphs( X_test,y_test,200,limit)
pixals_counter=create_pixals_counter2(list_of_graphs)

print("\nnaive bayes\n")
#naive bayes
temp=copy.deepcopy(pixals_counter)
matrix_result=run_test(temp,img_test,test_labels,number_of_images)
print_matrix(matrix_result,number_of_images,results_file)


print("\nKNN\n")
### KNN 
clf = KNeighborsClassifier(n_neighbors=5,algorithm='auto',n_jobs=10)
clf.fit(X_train,y_train)
##Traning
y_pred = clf.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
conf_mat = confusion_matrix(y_train,y_pred)
print('\nConfusion Matrix: \n',conf_mat)
##Test
test_labels_pred = clf.predict(test_img)
acc = accuracy_score(test_labels,test_labels_pred)
conf_mat_test = confusion_matrix(test_labels,test_labels_pred)
print("conf_mat_test")
print('\nConfusion Matrix for Test Data: \n',conf_mat_test)

print("\nLBP\n")
#LBP
label=bp(temp,list_of_graphs[0][0])
for cnt_loop in range(number_of_loops):
    label=bp2(label,list_of_graphs[0][0])
    label=bp(label,list_of_graphs[0][0])
    matrix_result=run_test(label,img_test,test_labels,number_of_images)
    print("number of loops=" + str(cnt_loop+1))
    print_matrix(matrix_result,number_of_images,results_file)


print("\ncreative part\n")
# creative part
temp_image_test=copy.deepcopy(img_test)
image_binary(temp_image_test,limit)
for i in range(1,9):
        print("number of edges="+ str(number_of_edges*i*2))
        matrix=run_test2(upper_bound,low_bound,temp_image_test,test_labels,number_of_images,pixals_counter,limit,number_of_edges*i)
        print_matrix(matrix,number_of_images,results_file)


sys.stdout = old_stdout
results_file.close()
matan=1
