from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, RepeatedKFold
import pandas as pd
from DataPreProcessing import DataPreProcessing
from sklearn.metrics import SCORERS, confusion_matrix, roc_auc_score
import numpy as np
import pickle
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


'''
https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
https://scikit-learn.org/stable/modules/model_evaluation.html
https://scikit-learn.org/stable/modules/cross_validation.html
https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation-in-python-r/
https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
https://www.ritchieng.com/machine-learning-project-boston-home-prices/
https://www.datacamp.com/community/tutorials/categorical-data
https://www.analyticsvidhya.com/blog/2015/11/easy-methods-deal-categorical-variables-predictive-modeling/
'''

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def getDataSet():
    config_path = "./config/config"
    data = pd.read_csv("./AllData.csv")
    print(data.head(5))
    data_preprocessed = DataPreProcessing(data, config_path)
    retention = data_preprocessed.get_pre_processed_data()
    
    #numerical = form.getClassifierNumericalColumns('millenial')
    numerical = data_preprocessed.get_numerical()
    #categorical = form.getClassifierCatagoricalColumns('millenial')
    categorical = data_preprocessed.get_categorical()
    categorical.remove('Attrition')
    # Store the categorical data in a dataframe called attrition_cat
    attrition_cat = retention[categorical]
    
    #One hot vector for categorical columns
    attrition_cat = pd.get_dummies(attrition_cat)
    #print(attrition_cat.head(3))
    
    # Store the numerical features to a dataframe attrition_num
    attrition_num = retention[numerical]
    # Concat the two dataframes together columnwise
    features = pd.concat([attrition_num, attrition_cat], axis=1)
    # Define a dictionary for the target mapping
    print(retention["Attrition"].head(5))
    # Use the pandas apply method to numerically encode our attrition target variable
    #target_map = {'Yes':1, 'No':0}        
    #y = retention["Attrition"].apply(lambda x: target_map[x])
    labels = retention["Attrition"].apply(lambda x: int(x))
    
    return (features, labels)

def selectClassifier(features, labels, CV, scoring_pattern):
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0),
        MultinomialNB(),
        LogisticRegression(random_state=0)
    ]
    ''' Get all 'scoring' values '''
    print(sorted(SCORERS.keys()))
    #cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    modelAverage = []
    for model in models:
        avg = 0.0
        count = 0
        model_name = model.__class__.__name__
        #print(model_name)
        accuracies = cross_val_score(estimator=model, X=features, y=labels, scoring=scoring_pattern, cv=CV)
        #print(accuracies)
        for fold_idx, accuracy in enumerate(accuracies):
            avg = avg + accuracy
            count = count + 1
            entries.append((model_name, fold_idx, accuracy))
        avg = avg/count
        modelAverage.append((model, model_name, avg))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', scoring_pattern])
    print(cv_df)
    #print(modelAverage)
    '''
    Select classifier
    '''
    avgAcc = 0
    modelSelection = ()
    for modelAvg in modelAverage:
        if(avgAcc < modelAvg[2]):
            avgAcc = modelAvg[2]
            modelSelection = modelAvg 
    print('Selected Model='+str(modelSelection[1])+', Avg Score='+str(modelSelection[2]))
    

def confusionMatrix(features, labels, seed):
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0),
        MultinomialNB(),
        LogisticRegression(random_state=0)
    ]
    
    for model in models:
        print('*****************' + model.__class__.__name__ + '**********************')
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_score =[]
        i=1        
        
        for train_index,test_index in kf.split(features, labels):
            print('{} of KFold {}'.format(i,kf.n_splits))
            xtr,xvl = features.loc[train_index], features.loc[test_index]
            ytr,yvl = labels.loc[train_index], labels.loc[test_index]
            #model
            model.fit(xtr,ytr)
            score = roc_auc_score(yvl, model.predict(xvl))
            print('ROC AUC score:',score)
            cv_score.append(score)    
            i+=1
        print('Confusion matrix\n',confusion_matrix(yvl, model.predict(xvl)))
        print('Cv', cv_score,'\nMean cv Score', np.mean(cv_score))
        
        filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))
    
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.score(xvl, yvl)
        print(xvl.head(5))
        print('Results from loaded model', result)
        
        # Compute predictions/get employees
        employees = []
        predictedData = []
        prediction = loaded_model.predict(xvl)
        for i, empId in enumerate(xvl["ID"]):
            if(prediction[i] == 1):
                employees.append(empId)
                predictedData.append(xvl.iloc[i])
                #print("-->", i, id)
        print(len(employees))
        print(len(predictedData))
        #print(predictedData)
        predictedDataPd = pd.DataFrame(predictedData) 
        #predictedDataPd = pd.get_dummies(predictedDataPd)
        print(predictedDataPd.head(5))
        X = StandardScaler().fit_transform(predictedDataPd)
        #clusterDBScan(X)
        #clusterKMeans(X)
        
def clusterKMeans(X):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);    
            
def clusterDBScan(X):
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN().fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(labels)
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
#     print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#     print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#     print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#     print("Adjusted Rand Index: %0.3f"
#           % metrics.adjusted_rand_score(labels_true, labels))
#     print("Adjusted Mutual Information: %0.3f"
#           % metrics.adjusted_mutual_info_score(labels_true, labels,
#                                                average_method='arithmetic'))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    
    # #############################################################################
    # Plot result    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
    
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
        

def execute():
    data = getDataSet()
    features = data[0]
    labels = data[1]
#     selectClassifier(features, labels, 5, 'precision')
#     selectClassifier(features, labels, KFold(n_splits=5, shuffle=False), 'recall')
#     selectClassifier(features, labels, RepeatedKFold(n_splits=5, n_repeats=5), 'precision')
#     selectClassifier(features, labels, StratifiedKFold(n_splits=5, shuffle=False), 'precision')
    confusionMatrix(features, labels, 45)

execute()   