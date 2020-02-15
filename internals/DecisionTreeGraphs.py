from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
from com.sap.hr.retention.DataLoader import DataLoader
import pandas as pd
import os
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
from wand.api import library
import wand.color
import wand.image

'''
https://github.com/parrt/dtreeviz
https://github.com/parrt/dtreeviz/blob/master/notebooks/examples.ipynb
http://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-windows
1.Install windows package from: https://graphviz.gitlab.io/_pages/Download/Download_windows.html 
2.pip install graphviz, pip install dtreeviz
3.Add C:\Program Files (x86)\Graphviz2.38\bin to User path
4.Add C:\Program Files (x86)\Graphviz2.38\bin\dot.exe to System Path
'''
class DecisionTree(object):
    def __init__(self, ageType, dataFile):
        self.dataLoader = DataLoader(dataFile + ageType + '.csv')
        self.retention = self.dataLoader.getData()
        print(self.retention)
        self.type = ageType
        os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        self.initialize()
        
    
    def initialize(self):
        # load data
        # split data into X and y
        numerical = self.dataLoader.getClassifierNumericalColumns(self.type)
        
        categorical = self.dataLoader.getClassifierCatagoricalColumns(self.type)
        # Store the categorical data in a dataframe called attrition_cat
        attrition_cat = self.retention[categorical]
        
        ''' One hot vector for categorical columns '''
        attrition_cat = pd.get_dummies(attrition_cat)
        '''print(attrition_cat.head(3))'''
        
        # Store the numerical features to a dataframe attrition_num
        attrition_num = self.retention[numerical]
        # Concat the two dataframes together columnwise
        self.X = pd.concat([attrition_num, attrition_cat], axis=1)
        print(self.X.head(4))
        # Define a dictionary for the target mapping
        target_map = {'Yes':1, 'No':0}
        # Use the pandas apply method to numerically encode our attrition target variable
        self.y = self.retention["Attrition"].apply(lambda x: target_map[x])
        print(self.y.head(4))        

    def generateGraph1(self):
        # fit model no training data
        model = XGBClassifier()
        model.fit(self.X, self.y)
        # plot single tree
        plot_tree(model)
        #plot_tree(model, num_trees=0, rankdir='LR')
        fig = plt.gcf()
        fig.set_size_inches(150, 100)
        fig.savefig('dt1.png')        
        #plt.show()

    def generateGraph2(self):
        regr = tree.DecisionTreeRegressor(max_depth=4)
        regr.fit(self.X, self.y)
        
        viz = dtreeviz(regr,
                       self.X,
                       self.y,
                       target_name='Attrition',
                       feature_names=self.X.columns.values)
        
        svg_code = viz.svg()
        self.saveImage(svg_code.encode('utf8'), 'dt2.png')
        #viz.view()        

    def generateGraph3(self):
        regr = tree.DecisionTreeClassifier(max_depth=4)
        regr.fit(self.X, self.y)
        print(dict(zip(self.retention.columns, regr.feature_importances_)))
        print(self.X.columns.values)
        
        viz = dtreeviz(regr,
                       self.X,
                       self.y,
                       target_name='Attrition',
                       feature_names=self.X.columns.values,
                       class_names=["yes", "no"])
        
        svg_code = viz.svg()
        self.saveImage(svg_code.encode('utf8'), 'dt3.png')
        #viz.view()        
    
    def generateGraph4(self):
        regr = tree.DecisionTreeClassifier(max_depth=4)
        regr.fit(self.X, self.y)
        #Random Single observation for prediction.        
        X = self.X.iloc[np.random.randint(0, len(self.X)),::]  # random sample from training
        print(X)
        
        viz = dtreeviz(regr,
                       self.X,
                       self.y,
                       target_name='Attrition',
                       orientation ='LR',  # left-right orientation                       
                       feature_names=self.X.columns.values,
                       class_names=["yes", "no"],
                       X=X) # need to give single observation for prediction
        
        svg_code = viz.svg()
        self.saveImage(svg_code.encode('utf8'), 'dt4.png')
        #viz.view()
    
    def generateGraph5(self):
        classifier = tree.DecisionTreeClassifier(max_depth=4)  # limit depth of tree
        classifier.fit(self.X, self.y)

        viz = dtreeviz(classifier,
                       self.X, 
                       self.y,
                       target_name='Attrition',
                       feature_names=self.X.columns.values, 
                       class_names=["yes", "no"],
                       fancy=False )  # fance=False to remove histograms/scatterplots from decision nodes
            
        svg_code = viz.svg()
        self.saveImage(svg_code.encode('utf8'), 'dt5.png')
        #viz.view()
    
    def generateGraph6(self):
        clas = tree.DecisionTreeClassifier(max_depth=4)  
        clas.fit(self.X, self.y)
        
        # "8x8 image of integer pixels in the range 0..16."
        # columns = [f'pixel[{i},{j}]' for i in range(len(self.X.columns.values)) for j in range(len(self.X.columns.values))]
        
        
        viz = dtreeviz(clas, 
                       self.X, 
                       self.y,
                       target_name='Attrition',
                       feature_names=self.X.columns.values, 
                       class_names=["yes", "no"],
                       histtype='bar', 
                       orientation ='TD')
        svg_code = viz.svg()
        self.saveImage(svg_code.encode('utf8'), 'dt6.png')
        #viz.view()

     
    def generateGraph7(self):
        '''
            ['YearsSinceLastPromotion' 'TrainingHoursLastYear' 'TotalWorkingYears'
             'PerformanceRating' 'MonthlyIncome' 'PercentSalaryHike' 'Gender_Female'
             'Gender_Male' 'MaritalStatus_Married' 'MaritalStatus_Single'
             'Education_Graduate' 'Education_Under-Graduate']        
        '''
        #X = self.X.drop('TotalWorkingYears', axis=1)
        figsize = (6, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        features = [2, 4]
        X = self.X.values[:, features]
        print(X)
        y = self.X['TotalWorkingYears']
        t = rtreeviz_bivar_heatmap(ax,
                                   X, 
                                   y,
                                   max_depth=4,
                                   feature_names=['TotalWorkingYears', 'MonthlyIncome'],
                                   fontsize=14)
        plt.show()
        ''' 
        fig = plt.figure()
        ax = fig.gca()
        t = rtreeviz_bivar_3D(ax,
                    self.X['salaryHike'], 
                    self.y['monthlyIncome'],
                    max_depth=4,
                    feature_name=['SalaryHike', 'monthlyIncome'],
                    target_name='Attrition',
                    fontsize=14,
                    elev=20,
                    azim=25,
                    dist=8.2,
                    show={'splits','title'})
        plt.show()
        '''

    def saveImage(self, svgBytes, fileName):
        with wand.image.Image() as image:
            with wand.color.Color('transparent') as background_color:
                library.MagickSetBackgroundColor(image.wand, background_color.resource) 
            image.read(blob=svgBytes)
            png_image = image.make_blob("png32")

        with open(fileName, "wb") as out:
            out.write(png_image)
            
if __name__ == '__main__':
    dt = DecisionTree('millenial', '../input/')
#     dt.generateGraph1()
#     dt.generateGraph2()
    dt.generateGraph3()
#     dt.generateGraph4()
#     dt.generateGraph5()
#     dt.generateGraph6()
#     dt.generateGraph7()
