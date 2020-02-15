import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from IPython.display import Image as PImage
import pydot
from PIL import Image, ImageDraw, ImageFont
from DataLoader import DataLoader
import csv
from sklearn import metrics
import numpy as np

class DecisionPath(object):

    def __init__(self, dataFile):
        self.dataLoader = DataLoader(dataFile)
        self.retention = self.dataLoader.getData()
        print(self.retention.head(4))
        self.userDetails = {}
    
    def constructUserMap(self, employeeIdKey, managerReasonKey, exitReasonKey):
        index = 0
        for key in self.retention[employeeIdKey]:
            managerReason = self.retention[managerReasonKey][index]
            if(str(managerReason) == 'nan'):
                managerReason = ""
            exitReason = self.retention[exitReasonKey][index]                
            if(str(exitReason) == 'nan'):
                exitReason = ""                    
            valueTuple = (managerReason, exitReason)
            self.userDetails[key] = valueTuple
            index = index + 1

    def saveDecisionPath(self, maxDepth, outputFileLoc, type, employeeIdPos, employeeIdKey, managerReasonKey, exitReasonKey):
        ''' Construct user details map '''
        self.constructUserMap(employeeIdKey, managerReasonKey, exitReasonKey)

        # Refining our list of numerical variables        
        numerical = self.dataLoader.getClassifierNumericalColumns(type)
        
        categorical = self.dataLoader.getClassifierCatagoricalColumns(type)

        # Store the categorical data in a dataframe called attrition_cat
        attrition_cat = self.retention[categorical]
        '''attrition_cat = attrition_cat.drop(['Attrition'], axis=1) # Dropping the target column'''   
        
        attrition_cat = pd.get_dummies(attrition_cat)
        '''print(attrition_cat.head(3))'''
        
        # Store the numerical features to a dataframe attrition_num
        attrition_num = self.retention[numerical]
        # Concat the two dataframes together columnwise
        attrition_final = pd.concat([attrition_num, attrition_cat], axis=1)
        '''print(attrition_final.head(4))'''
        
        # Define a dictionary for the target mapping
        target_map = {'Yes':1, 'No':0}
        # Use the pandas apply method to numerically encode our attrition target variable
        target = self.retention["Attrition"].apply(lambda x: target_map[x])
        '''print(target.head(3))'''
        
        # Split data into train and test sets as well as for validation and testing
        X_train, X_test, Y_train, Y_test = train_test_split(attrition_final, target, random_state=0);
        #train, test, target_train, target_val = StratifiedShuffleSplit(attrition_final, target, random_state=0);
        
        print("Test Data")
        testArray = attrition_final.iloc[:,:].values
        '''print(testArray)'''
        testTargetArray = target.iloc[:].values
        '''print(testTargetArray)'''
        oversampler = SMOTE(random_state=0)
        smote_train, smote_target = oversampler.fit_sample(X_train, Y_train)
        
        seed = 0   # We set our random seed to zero for reproducibility
        # Random Forest parameters
        rf_params = {
            'n_jobs': -1,
            'n_estimators': 800,
            'warm_start': True, 
            'max_features': 0.3,
            'max_depth': 9,
            'min_samples_leaf': 2,
            'max_features' : 'sqrt',
            'random_state' : seed,
            'verbose': 0
        }
        #estimator = RandomForestClassifier(**rf_params)
        #estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
        estimator = DecisionTreeClassifier(max_leaf_nodes=7, random_state=0)        

        estimator.fit(smote_train, smote_target)
        print("Fitting of Decision Tree as finished")
        
        Y_pred = estimator.predict(X_test)
        print("Decision tree predictions finished")
        
        accScore = accuracy_score(Y_test, Y_pred)
        print("Accuracy=%s" %accScore)

        print("Mean Absolute Error:", metrics.mean_absolute_error(Y_test, Y_pred))
        print("Mean Squared Error:", metrics.mean_squared_error(Y_test, Y_pred))
        print("Root Mean Absolute Error:", np.sqrt(metrics.mean_absolute_error(Y_test, Y_pred)))

        # Export our trained model as a .dot file
        with open(outputFileLoc + type + '_Decision_Tree.dot', 'w') as f:
             f = tree.export_graphviz(estimator,
                                      out_file=f,
                                      max_depth = maxDepth,
                                      impurity = False,
                                      feature_names = X_train.columns,
                                      class_names = ['No', 'Yes'],
                                      rounded = True,
                                      node_ids=1,
                                      filled= True )
                
        #Convert .dot to .png to allow display in web notebook
        #check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])
        
        (graph,) = pydot.graph_from_dot_file(outputFileLoc + type + '_Decision_Tree.dot')
        graph.write_png(outputFileLoc + type + '_Decision_Tree.png')
        
        # Annotating chart with PIL
        '''
        img = Image.open(outputFileLoc + type + '_Decision_Tree.png')
        draw = ImageDraw.Draw(img)
        img.save(outputFileLoc + type + '_DecisionTree_sample-out.png')
        PImage(outputFileLoc + type + '_DecisionTree_sample-out.png')
        '''

        # The decision estimator has an attribute called tree_  which stores the entire
        # tree structure and allows access to low level attributes. The binary tree
        # tree_ is represented as a number of parallel arrays. The i-th element of each
        # array holds information about the node `i`. Node 0 is the tree's root. 
        # NOTE:
        # Some of the arrays only apply to either leaves or split nodes, resp. In this
        # case the values of nodes of the other type are arbitrary!
        #
        # Among those arrays, we have:
        #   - left_child, id of the left child of the node
        #   - right_child, id of the right child of the node
        #   - feature, feature used for splitting the node
        #   - threshold, threshold value at the node
        
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        featureNames  = [X_train.columns[i] for i in estimator.tree_.feature]
        threshold = estimator.tree_.threshold
        
        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
        
            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        
        print("The binary tree structure has %s nodes and has "
              "the following tree structure:"
              % n_nodes)
        for i in range(n_nodes):
            if is_leaves[i]:
                print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            else:
                print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                      "node %s."
                      % (node_depth[i] * "\t",
                         i,
                         children_left[i],
                         feature[i],
                         threshold[i],
                         children_right[i],
                         ))
        print()
        
        # First let's retrieve the decision path of each sample. The decision_path
        # method allows to retrieve the node indicator functions. A non zero element of
        # indicator matrix at the position (i, j) indicates that the sample i goes
        # through the node j.
        
        node_indicator = estimator.decision_path(testArray)
        
        # Similarly, we can also have the leaves ids reached by each sample.
        
        leave_id = estimator.apply(testArray)
        '''print(leave_id)'''
        
        # Now, it's possible to get the tests that were used to predict a sample or
        # a group of samples. First, let's make it for the sample.
        count = 0
        # Define a dictionary for the target mapping
        target_reverse_map = {1:'Yes', 0:'No'}
        
        rows = [['UserId', 'Nodes', 'Path', 'Names', 'Values', managerReasonKey, exitReasonKey]]
        
        '''Get all user id's '''
        for index in range(len(testArray)):
            nodeNames = ""
            nodePath = ""
            nodeValues = ""
            ''' 0 th position is employee number '''
            userId = self.retention[employeeIdKey][index]
            #userId = testArray[index][employeeIdPos]
            node_index = node_indicator.indices[node_indicator.indptr[index]:
                                                node_indicator.indptr[index + 1]]
            
            '''print('Rules used to predict user id: %s' % (userId))'''
            '''print(node_index)'''
            for node_id in node_index:
                if leave_id[index] == node_id:  # <-- changed != to ==
                    #continue # <-- comment out
                    nodePath = nodePath + "leaf node " + str(leave_id[index]) + " (" + target_reverse_map[testTargetArray[index]] +")"
                    nodeNames =  nodeNames + target_reverse_map[testTargetArray[index]]
                    nodeValues = nodeValues + "leaf node " + str(leave_id[index]) + " (" + target_reverse_map[testTargetArray[index]] +")"                    
                    '''print("leaf node {} reached".format(leave_id[index])) # <--'''
            
                else: # < -- added else to iterate through decision nodes
                    if (testArray[index, feature[node_id]] <= threshold[node_id]):
                        threshold_sign = "<="
                    else:
                        threshold_sign = ">"
                    nodePath = nodePath + "node "+str(node_id) + "-->"
                    nodeNames =  nodeNames + featureNames[node_id] + "-->"
                    nodeValues = nodeValues + "node "+str(node_id) + ": (X["+str(index)+","+str(feature[node_id])+"] (= "+str(testArray[index, feature[node_id]])+") "+threshold_sign + " " +str(threshold[node_id])+")-->"                                                                                                                   
                    '''print("decision id node %s : (X[%s, %s] (= %s) %s %s)"
                          % (node_id,
                             index,
                             feature[node_id],
                             testArray[index, feature[node_id]], # <-- changed i to rowId
                             threshold_sign,
                             threshold[node_id]))'''
            count = count + 1            
            empReasonTuple = self.userDetails[userId]
            rows.append([userId, node_index, nodePath, nodeNames, nodeValues, empReasonTuple[0], empReasonTuple[1]])
            '''print("----------------------------------------------------------")'''

        print("Total=%s"%count)
        with open(outputFileLoc + type + '_user_path.csv', 'w') as csvFile:
            wr = csv.writer(csvFile)
            wr.writerows(rows)
        csvFile.close()

if __name__ == '__main__':
    milenialDecisionPath = DecisionPath('../../../../input/millenial.csv')
    milenialDecisionPath.saveDecisionPath(5, '../../../../output/', 'millenial', 0, 'ID', 'ReasonToManager', 'ReasonInExitInterview')
    
    decisionPath = DecisionPath('../../../../input/genx.csv')
    decisionPath.saveDecisionPath(5, '../../../../output/', 'genx', 0, 'ID', 'ReasonToManager', 'ReasonInExitInterview')
    
    decisionPath = DecisionPath('../../../../input/babyboomer.csv')
    decisionPath.saveDecisionPath(5, '../../../../output/', 'babyboomer', 0, 'ID', 'ReasonToManager', 'ReasonInExitInterview')