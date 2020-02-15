from flask import Flask, request, abort
import json
import pandas as pd
from DataPreProcessing import DataPreProcessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import SCORERS, roc_auc_score, accuracy_score
import pickle
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from kmodes.kprototypes import KPrototypes

DEBUG = True
app = Flask(__name__)
app.debug = DEBUG
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
MODEL_FILE = 'finalized_model.sav'
CLASS_LABEL_FILE = 'classLabels_mapping.sav'
FINAL_RESULT_FILE = 'final.csv'

def getInputDataSet(data, config_path, classLabel):
    data_preprocessed = DataPreProcessing(data, config_path, classLabel)
    retention = data_preprocessed.get_pre_processed_data()
    print(retention)
    #numerical = form.getClassifierNumericalColumns('millenial')
    numerical = data_preprocessed.get_numerical()
    print(numerical)
    
    #categorical = form.getClassifierCatagoricalColumns('millenial')
    categorical = data_preprocessed.get_categorical()
    print(categorical)
    categorical.remove(classLabel)
    # Store the categorical data in a dataframe called attrition_cat
    attrition_cat = retention[categorical]
    
    #One hot vector for categorical columns
    attrition_cat = pd.get_dummies(attrition_cat)
    #print(attrition_cat.head(3))
    
    # Store the numerical features to a dataframe attrition_num
    attrition_num = retention[numerical]
    # Concat the two dataframes together columnwise
    features = pd.concat([attrition_num, attrition_cat], axis=1)
    # Remove class label from features
    print(features)
    # Define a dictionary for the target mapping
    print(retention[classLabel].head(5))
    # Use the pandas apply method to numerically encode our attrition target variable
    #target_map = {'Yes':1, 'No':0}        
    #y = retention["Attrition"].apply(lambda x: target_map[x])
    labels = retention[classLabel].apply(lambda x: int(x))
    print(labels)
    classMapping = data_preprocessed.getClassLabelMapping()
    return (features, labels, classMapping)

def getPredictionDataSet(data, config_path):
    data_preprocessed = DataPreProcessing(data, config_path, None)
    retention = data_preprocessed.get_pre_processed_data()
    print(retention)
    #numerical = form.getClassifierNumericalColumns('millenial')
    numerical = data_preprocessed.get_numerical()
    print(numerical)
    
    #categorical = form.getClassifierCatagoricalColumns('millenial')
    categorical = data_preprocessed.get_categorical()
    print(categorical)
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
    print(features)
    # Use the pandas apply method to numerically encode our attrition target variable
    #target_map = {'Yes':1, 'No':0}        
    #y = retention["Attrition"].apply(lambda x: target_map[x])
    return features

def getParsedData(data, isRequireClassLabel):
    jsonObj = json.loads(data)
    classLabel = ""
    if(isRequireClassLabel):
        classLabel = jsonObj['classLabel']
    print(classLabel)
    jsonEntity = jsonObj['data']
    print('Length of data', len(jsonEntity))    
    headerCount = 0
    header = ""
    parsedStr = ""    
    for entity in jsonEntity:
        count = 0
        for prop in entity:
            if(headerCount == 0):
                header = header + prop
            parsedStr = parsedStr + str(entity[prop])
            if(count < len(entity)-1):
                parsedStr = parsedStr + ','
                if(headerCount == 0):
                    header = header + ','
            count += 1
        parsedStr = parsedStr + "\n"
        headerCount += 1
    parsedStr = header + "\n" + parsedStr
    print(parsedStr)
    return (classLabel, parsedStr)


def selectClassifier(features, labels, seed):
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0),
        MultinomialNB(),
        LogisticRegression(random_state=0)
    ]
    
    modelAverage = []
    for model in models:
        model_name = model.__class__.__name__
        print('*****************' + model_name + '**********************')
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_score =[]
        i=1        
        
        for train_index,test_index in kf.split(features, labels):
            print('{} of KFold {}'.format(i, kf.n_splits))
            xtr,xvl = features.loc[train_index], features.loc[test_index]
            ytr,yvl = labels.loc[train_index], labels.loc[test_index]
            #model
            model.fit(xtr,ytr)
            score = accuracy_score(yvl, model.predict(xvl))
            print('Accuracy score:',score)
            cv_score.append(score)    
            i+=1
        #print('Confusion matrix\n',confusion_matrix(yvl, model.predict(xvl)))
        avg = np.mean(cv_score)
        print('Cv', cv_score,'\nMean cv Score', np.mean(cv_score))
        modelAverage.append((model, model_name, avg))
        
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
    return modelSelection
        
@app.route('/sap/upload', methods=['POST'])
def uploadData():
    if not request.json:
        abort(400)
    print(request.json)
    jsonData = json.dumps(request.json)
    ''' Parse JSON and get employee data '''
    parsedData = getParsedData(jsonData, True)
    classLabel = parsedData[0]
    file = open('dtanalysis.csv', 'w')
    file.write(parsedData[1])
    file.close()
    
    config_path = "./config/config"
    data = pd.read_csv("./dtanalysis.csv")
    inputData = getInputDataSet(data, config_path, classLabel)
    features = inputData[0]
    labels = inputData[1]
    print('Length of upload data after parsing', len(features))    
    modelSelection = selectClassifier(features, labels, 45)
    model = modelSelection[0]
    modelName = modelSelection[1]
    accuracy = modelSelection[2]
    pickle.dump(model, open(MODEL_FILE, 'wb'))
    pickle.dump(inputData[2], open(CLASS_LABEL_FILE, 'wb'))
    
    resultJson = {}
    resultJson["success"] = True
    resultJson["modelname"] = modelName
    resultJson["modelaccuracy"] = accuracy * 100
    jsonData = json.dumps(resultJson)
    return jsonData, 200, {'ContentType':'application/json'}

@app.route('/sap/predict', methods=['POST'])
def performPrediction():
    if not request.json:
        abort(400)
    print(request.json)
    jsonData = json.dumps(request.json)
    ''' Parse JSON and get employee data '''
    parsedData = getParsedData(jsonData, False)
    ''' Construct dataframe '''
    file = open('prediction.csv', 'w')
    file.write(parsedData[1])
    file.close()
    
    config_path = "./config/config"    
    data = pd.read_csv("./prediction.csv")
    orgData = data.copy()
    features = getPredictionDataSet(data, config_path)
    print('Length of data after parsing prediction request', len(features))
    
    # load the model from disk
    loaded_model = pickle.load(open(MODEL_FILE, 'rb'))
    loaded_classLabel = pickle.load(open(CLASS_LABEL_FILE, 'rb'))
    
    # Compute predictions/get employees
    prediction = loaded_model.predict(features)
    print(prediction)
    present = 0
    absent = 0 
    for pred in prediction:
        if(pred == 1):
            absent+=1
        else:
            present+=1
    print("absent", absent)
    print("present", present)
    
    predictionDf = pd.DataFrame({'prediction':prediction})
    predictionDf['prediction'] = predictionDf['prediction'].apply(lambda x: loaded_classLabel[x])    
    df = pd.concat([predictionDf, orgData], axis=1)
    ''' Response to csv file '''
    df.to_csv(FINAL_RESULT_FILE, encoding='utf-8', index=False)
    outputJson = df.to_json(orient = "records")
    return outputJson, 200, {'ContentType':'application/json'}

@app.route('/sap/predict/cluster', methods=['POST'])
def performPredictionCluster():
    if not request.json:
        abort(400)
    print(request.json)
    jsonData = json.dumps(request.json)
    jsonObj = json.loads(jsonData)
    classLabel = jsonObj['classlabel']
    classValue = jsonObj['classvalue']
    employeeID = jsonObj['id']
    
    if 'noofclusters' not in jsonObj:
        noOfClusters = 3
    else:
        noOfClusters = jsonObj['noofclusters']    
  
    allData = pd.read_csv("./final.csv")
    originalData = allData.copy()
    
    ''' Split the data in to separate file based on class label Ex: YES/NO for analysis''' 
    for part_id, df_id in allData.groupby(classLabel):
        df_id.to_csv(f'final_{part_id}.csv',index=False)    
    
    classFilename = "./final_" + classValue + ".csv"
    data = pd.read_csv(classFilename)
    ''' Drop class label and employee id column '''
    data.drop([classLabel, employeeID], axis = 1, inplace = True)     
    print(data.dtypes)
    print(data)
    
    colIndex = 0
    numericalColNames = []
    categoricalColNames = []
    categoricalColIndex = []
    
    for col in data.columns:
        if (is_string_dtype(data[col])):
            categoricalColNames.append(col)
            categoricalColIndex.append(colIndex)
        elif (is_numeric_dtype(data[col])):
            numericalColNames.append(col)
        colIndex = colIndex + 1  

    print(numericalColNames)
    print(categoricalColNames)

    kproto = KPrototypes(n_clusters=noOfClusters, init='Cao', verbose=2)
    clusters = kproto.fit_predict(data, categorical=categoricalColIndex)
    
    # Print cluster centroids of the trained model.
    print(kproto.cluster_centroids_)
    # Print training statistics
    print(kproto.cost_)
    print(kproto.n_iter_)
    print(kproto.labels_)
    
    '''Add all the data back'''
    data['cluster'] = clusters
    removedAttributes = originalData[[classLabel, employeeID]]
    resultData = pd.concat([removedAttributes, data], axis=1)
    sortedClusteredData = resultData.sort_values('cluster')  
    '''sort data based on cluster numbers'''
    outputJson = sortedClusteredData.to_json(orient = "records")
    
    ''' remove class label and empoyeeID for computing clustered mode '''
    sortedClusteredData = data.sort_values('cluster') 
    ''' Get column mode '''
    clusteredGroupDfMap = {}
    for clusterNo in range(noOfClusters):
        clusteredGroupDf = {}
        ''' Get clustered group data '''
        clusteredDf = sortedClusteredData.loc[sortedClusteredData['cluster'] == clusterNo]
        ''' Get mode for all columns '''
        for columnName in list(clusteredDf):
            columnData = clusteredDf[columnName]
            #groupedByColumn = columnData.groupby([columnName]).count()
            groupedByColumn = clusteredDf.groupby(columnName)[columnName].transform('count')
            clusteredGroupDf[columnName] = groupedByColumn
        clusteredGroupDfMap[clusterNo] = clusteredGroupDf
    
    
    return outputJson, 200, {'ContentType':'application/json'}
    
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)