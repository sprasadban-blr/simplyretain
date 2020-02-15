from DataLoader import DataLoader
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go

class DataAnalysis(object):
    def __init__(self, type, dataFile):
        self.type = type
        self.dataLoader = DataLoader(dataFile + type + '.csv')
        self.retention = self.dataLoader.getData()
        
    def generatePlots(self):
        # Plotting the KDEplots        
        f, axes = plt.subplots(3, 3, figsize=(10, 8), 
                               sharex=False, sharey=False)
        
        # Defining our colormap scheme
        s = np.linspace(0, 3, 10)

        cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)

        # Generate and plot
        x = self.retention['YearsWithCurrentManager'].values
        y = self.retention['YearsSinceLastPromotion'].values
        sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=axes[0,0])
        axes[0,0].set( title = 'Yearswithcurrentmanager Vs Yearssincelastpromotion')
        
        cmap = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)
        # Generate and plot
        x = self.retention['YearsWithCurrentManager'].values
        y = self.retention['PercentSalaryHike'].values
        sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,1])
        axes[0,1].set( title = 'Yearswithcurrentmanager Vs Percentsalaryhike')

        cmap = sns.cubehelix_palette(start=0.666666666667, light=1, as_cmap=True)
        # Generate and plot
        x = self.retention['YearsSinceLastPromotion'].values
        y = self.retention['YearsInCurrentRole'].values
        sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,2])
        axes[0,2].set( title = 'Yearssincelastpromotion Vs Yearsincurrentrole')
        
        cmap = sns.cubehelix_palette(start=1.0, light=1, as_cmap=True)
        # Generate and plot
        x = self.retention['MonthlyIncome'].values
        y = self.retention['CommuteDistance'].values
        sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,0])
        axes[1,0].set( title = 'Monthlyincome Vs Commutedistance')
        
        cmap = sns.cubehelix_palette(start=1.333333333333, light=1, as_cmap=True)
        # Generate and plot
        x = self.retention['MonthlyIncome'].values
        y = self.retention['JobSatisfaction'].values
        sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,1])
        axes[1,1].set( title = 'Monthlyincome Vs Jobsatisfaction')
        
        cmap = sns.cubehelix_palette(start=1.666666666667, light=1, as_cmap=True)
        # Generate and plot
        x = self.retention['TotalWorkingYears'].values
        y = self.retention['JobSatisfaction'].values
        sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,2])
        axes[1,2].set( title = 'Totalworkingyears Vs Jobsatisfaction')
        
        cmap = sns.cubehelix_palette(start=2.0, light=1, as_cmap=True)
        # Generate and plot
        x = self.retention['TotalWorkingYears'].values
        y = self.retention['MonthlyIncome'].values
        sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,0])
        axes[2,0].set( title = 'Totalworkingyears Vs Monthlyincome')
        
        cmap = sns.cubehelix_palette(start=2.333333333333, light=1, as_cmap=True)
        # Generate and plot
        x = self.retention['JobSatisfaction'].values
        y = self.retention['YearsSinceLastPromotion'].values
        sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,1])
        axes[2,1].set( title = 'Jobsatisfaction Vs Yearssincelastpromotion')
        
        cmap = sns.cubehelix_palette(start=2.666666666667, light=1, as_cmap=True)
        # Generate and plot
        x = self.retention['WorkLifeBal'].values
        y = self.retention['JobSatisfaction'].values
        sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,2])
        axes[2,2].set( title = 'Worklifebalance Vs Jobsatisfaction')
        
        f.tight_layout()
        plt.show()
    
    def generateHeatMaps(self, outputFile):
                    # Define a dictionary for the target mapping
        target_map = {'Yes':1, 'No':0}
        # Use the pandas apply method to numerically encode our attrition target variable
        self.retention["Attrition_numerical"] = self.retention["Attrition"].apply(lambda x: target_map[x])

        # creating a list of only numerical values

        numerical = self.dataLoader.getClassifierNumericalColumns(self.type)
        '''print(attrition[numerical])'''
        
        data = [
            go.Heatmap(
                z= self.retention[numerical].astype(float).corr().values, # Generating the Pearson correlation
                x=self.retention[numerical].columns.values,
                y=self.retention[numerical].columns.values,
                colorscale='Viridis'
            )
        ]
        
        
        layout = go.Layout(
            title='Pearson Correlation of numerical features',
            xaxis = dict(ticks='', nticks=36),
            yaxis = dict(ticks='' ),
            width = 900, height = 700,
            
        )
        fig = go.Figure(data=data, layout=layout)
        fileName = outputFile + self.type + '_labelled-heatmap.html'
        py.plot(fig, filename=fileName)

if __name__ == '__main__':
    '''
    dataAnalysis = DataAnalysis('../../../../input/HR_Attrition.csv')
    dataAnalysis.generatePlots()
    dataAnalysis.generateHeatMaps('../../../../output/labelled-heatmap.html')
    '''
    
    dataAnalysis = DataAnalysis('millenial', '../../../../input/')
    dataAnalysis.generatePlots()
    dataAnalysis.generateHeatMaps('../../../../output/')    