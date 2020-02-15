''' https://www.datacamp.com/community/tutorials/categorical-data
    https://www.kaggle.com/questions-and-answers/55494 
    https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512
    (Cat2vec with Random Forest)
    https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b
'''
'''
1. Automatically detect categorical and continuous attributes
2. Automatically fill in missing values
3. Automatically identify correlated attributes influencing attrition
4. Build model, validate model 
5. Charts to depict the hidden insights 
'''
import pandas as pd;
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalysis(object):
    
    def __init__(self, fileLoc):
        df = pd.read_csv(fileLoc)
        ''' Data details '''
        print(df.head())
        print(df.tail())
        print(df.columns)
        print(df.shape)
        print(df.info())
        
        ''' Summary Statistics '''
        print('-----------------------------------------------')
        print('Summary Statistics....')
        print(df.describe());
        ''' Frequency count '''
        print('-----------------------------------------------')
        print('Frequency Counts.....')
        print(df.JobSatisfaction.value_counts(dropna=False))
        print('-----------------------------------------------')        
        print(df.JobTitle.value_counts(dropna=False))
        print('-----------------------------------------------')  
        print(df.Major.value_counts(dropna=False))
        print('-----------------------------------------------')  
        print(df.StockOption.value_counts(dropna=False))
        print('-----------------------------------------------')  
        print(df.Contentedness.value_counts(dropna=False))
        print('-----------------------------------------------')
        print(df.CommuteDistance.value_counts(dropna=False))
        print('-----------------------------------------------')
        print(df.Attrition.value_counts(dropna=False))
        print('-----------------------------------------------')
        ''' Check null values in any colums '''
        print(df.Attrition.isnull().sum())
        ''' Bar plots for discrete data counts, frequency distribution '''
        ''' Histogram for continuous data counts '''
        ''' Box plots for basic summary statistics to identify outliers '''
        ''' Scatter plots for relationship between 2 numerical values '''
        print('Histogram....')
        df.JobSatisfaction.plot('hist')
        plt.show()
        print('Box plot....')
        df.boxplot(column = 'CommuteDistance', by = 'Attrition', rot = 90)  
        plt.show()
        
        df_count = df['CommuteDistance'].value_counts()        
        sns.set(style="darkgrid")
        sns.barplot(df_count.index, df_count.values, alpha=0.9)
        plt.title('Frequency Distribution of Commute Distance')
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('CommuteDistance', fontsize=12)
        plt.show()
        
        retYes = df[df.Attrition == 'Yes']
        retYes['CommuteDistance'].plot('hist')
        plt.show()
        
        print('Scatter plot....')
        df.plot(kind='scatter', x='SalaryHike', y='PerfRating', rot=70)
        plt.show()        
        
        print('Bar Chart..')
        labels = df['Major'].astype('category').cat.categories.tolist()
        counts = df['Major'].value_counts()
        sizes = [counts[var_cat] for var_cat in labels]
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
        ax1.axis('equal')
        plt.show()
                
if __name__ == '__main__':
    analysis = DataAnalysis('../../../input/HR_Attrition.csv') 