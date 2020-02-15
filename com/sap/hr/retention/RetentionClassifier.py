import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#%matplotlib inline

# Import statements required for Plotly 
import plotly.offline as py
import plotly.graph_objs as go

# Import statements required for Plotly 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from imblearn.over_sampling import SMOTE
import xgboost

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import pydot
import re
from DataLoader import DataLoader
from sklearn import metrics
import numpy as np

# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')
class RetentionClassifier(object):
    def __init__(self, dataFile):
        self.dataLoader = DataLoader(dataFile)
        self.retention = self.dataLoader.getData()

    def classify(self, outputFileLoc, type):
        # Refining our list of numerical variables
        numerical = self.dataLoader.getClassifierNumericalColumns(type)
        
        categorical = self.dataLoader.getClassifierCatagoricalColumns(type)
        # Store the categorical data in a dataframe called attrition_cat
        attrition_cat = self.retention[categorical]
        
        ''' One hot vector for categorical columns '''
        attrition_cat = pd.get_dummies(attrition_cat)
        '''print(attrition_cat.head(3))'''
        
        # Store the numerical features to a dataframe attrition_num
        attrition_num = self.retention[numerical]
        # Concat the two dataframes together columnwise
        attrition_final = pd.concat([attrition_num, attrition_cat], axis=1)
        
        # Define a dictionary for the target mapping
        target_map = {'Yes':1, 'No':0}
        # Use the pandas apply method to numerically encode our attrition target variable
        target = self.retention["Attrition"].apply(lambda x: target_map[x])
        '''print(target.head(3))'''
        
        data = [go.Bar(
            x=self.retention["Attrition"].value_counts().index.values,
            y= self.retention["Attrition"].value_counts().values
        )]

        py.plot(data, filename=outputFileLoc+'bar-chart')
        
        # Split data into train and test sets as well as for validation and testing
        X_train, X_test, Y_train, Y_test = train_test_split(attrition_final, target, train_size= 0.75,random_state=0);
        #train, test, target_train, target_val = StratifiedShuffleSplit(attrition_final, target, random_state=0);
        
        oversampler=SMOTE(random_state=0)
        smote_X_train, smote_Y_train = oversampler.fit_sample(X_train, Y_train)
        
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
        rf = RandomForestClassifier(**rf_params)
        rf.fit(smote_X_train, smote_Y_train)
        print("Fitting of Random Forest as finished")
        
        rf_predictions = rf.predict(X_test)
        print("Random Forest predictions finished")
        
        accScore = accuracy_score(Y_test, rf_predictions)
        print("Accuracy=%s" %accScore)
        
        # Scatter plot 
        trace = go.Scatter(
            y = rf.feature_importances_,
            x = attrition_final.columns.values,
            mode='markers',
            marker=dict(
                sizemode = 'diameter',
                sizeref = 1,
                size = 13,
                #size= rf.feature_importances_,
                #color = np.random.randn(500), #set color equal to a variable
                color = rf.feature_importances_,
                colorscale='Portland',
                showscale=True
            ),
            text = attrition_final.columns.values
        )
        data = [trace]
        
        layout= go.Layout(
            autosize= True,
            title= 'Random Forest Feature Importance',
            hovermode= 'closest',
             xaxis= dict(
                 ticklen= 5,
                 showgrid=False,
                zeroline=False,
                showline=False
             ),
            yaxis=dict(
                title= 'Feature Importance',
                showgrid=False,
                zeroline=False,
                ticklen= 5,
                gridwidth= 2
            ),
            showlegend= False
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig,filename=outputFileLoc+'randomForest_scatterplot')
        
        decision_tree = tree.DecisionTreeClassifier(max_depth = 4)
        decision_tree.fit(X_train, Y_train)
        
        # Predicting results for test dataset
        Y_pred = decision_tree.predict(X_test)
        print("Mean Absolute Error:", metrics.mean_absolute_error(Y_test, Y_pred))
        print("Mean Squared Error:", metrics.mean_squared_error(Y_test, Y_pred))
        print("Root Mean Absolute Error:", np.sqrt(metrics.mean_absolute_error(Y_test, Y_pred)))
        
        # Export our trained model as a .dot file
        with open(outputFileLoc+'randomForest_tree.dot', 'w') as f:
            f = tree.export_graphviz(decision_tree,
                                      out_file=f,
                                      max_depth = 4,
                                      impurity = False,
                                      feature_names = attrition_final.columns.values,
                                      class_names = ['No', 'Yes'],
                                      rounded = True,
                                      filled= True )
                
        #Convert .dot to .png to allow display in web notebook
        #check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])
        
        (graph,) = pydot.graph_from_dot_file(outputFileLoc+'randomForest_tree.dot')
        graph.write_png(outputFileLoc+'randomForest_tree.png')
        
        # Annotating chart with PIL
        img = Image.open(outputFileLoc+"randomForest_tree.png")
        draw = ImageDraw.Draw(img)
        img.save(outputFileLoc+'randomForest_sample-out.png')
        PImage(outputFileLoc+"randomForest_sample-out.png")
        
        # Gradient Boosting Parameters
        gb_params ={
            'n_estimators': 500,
            'max_features': 0.9,
            'learning_rate' : 0.2,
            'max_depth': 11,
            'min_samples_leaf': 2,
            'subsample': 1,
            'max_features' : 'sqrt',
            'random_state' : seed,
            'verbose': 0
        }
        
        gb = GradientBoostingClassifier(**gb_params)
        # Fit the model to our SMOTEd train and target
        gb.fit(smote_X_train, smote_Y_train)
        # Get our predictions
        gb_Y_pred = gb.predict(X_test)
        print("Gradient boosting predictions finished")
        accScore = accuracy_score(Y_test, gb_Y_pred)
        print("Accuracy=%s" %accScore)
        print("Mean Absolute Error:", metrics.mean_absolute_error(Y_test, gb_Y_pred))
        print("Mean Squared Error:", metrics.mean_squared_error(Y_test, gb_Y_pred))
        print("Root Mean Absolute Error:", np.sqrt(metrics.mean_absolute_error(Y_test, gb_Y_pred)))
        
        # Scatter plot 
        trace = go.Scatter(
            y = gb.feature_importances_,
            x = attrition_final.columns.values,
            mode='markers',
            marker=dict(
                sizemode = 'diameter',
                sizeref = 1,
                size = 13,
                #size= rf.feature_importances_,
                #color = np.random.randn(500), #set color equal to a variable
                color = gb.feature_importances_,
                colorscale='Portland',
                showscale=True
            ),
            text = attrition_final.columns.values
        )
        data = [trace]
        
        layout= go.Layout(
            autosize= True,
            title= 'Gradient Boosting Model Feature Importance',
            hovermode= 'closest',
             xaxis= dict(
                 ticklen= 5,
                 showgrid=False,
                zeroline=False,
                showline=False
             ),
            yaxis=dict(
                title= 'Feature Importance',
                showgrid=False,
                zeroline=False,
                ticklen= 5,
                gridwidth= 2
            ),
            showlegend= False
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig,filename=outputFileLoc+'GradientBoosting_scatterplot')        
        return
    
if __name__ == '__main__':
    milRetention = RetentionClassifier('../../../../input/millenial.csv')
    milRetention.classify('../../../../output/millenial_', 'millenial')
    
    genRetention = RetentionClassifier('../../../../input/genx.csv')
    genRetention.classify('../../../../output/genx_', 'genx')
    
    babyBoomerRetention = RetentionClassifier('../../../../input/babyboomer.csv')
    babyBoomerRetention.classify('../../../../output/babyboomer_', 'babyboomer')
