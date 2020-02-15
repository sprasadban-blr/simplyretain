import pandas as pd

class DataLoader(object):
    def __init__(self, dataFile):
        self.inputFile = dataFile
        self.loadFile()
    
    def loadFile(self):
        self.attrition = pd.read_csv(self.inputFile)
        
    def getData(self):
        return self.attrition
    
    def getNumericColumns(self):
        numerical = [u'Age', u'DailyRate', u'CommuteDistance', u'Education', u'EmployeeNumber', 
                     u'Contentedness', u'HourlyRate', u'Participation', u'JobLevel', u'JobSatisfaction',
                     u'SalaryRate', u'NumPrevCompanies', u'SalaryHike', u'PerfRating', u'TeamBonding', 
                     u'StockOption', u'Experience', u'TrainingHours', u'WorkLifeBal', u'JoiningYear', 
                     u'CurrDeptYears', u'CurrPosYears', u'YearsWithCurrManager']
        return numerical
    
    def getCatagoricalColumns(self):
        categorical = [u'MaritalStatus', u'Gender', u'Department', 
                       u'JobTitle', u'Contentedness', u'JobSatisfaction']
        return categorical
    
    def getClassifierNumericalColumns(self, type):
        numerical = []
        if(type == 'millenial'):
            numerical = [u'YearsSinceLastPromotion', u'TrainingHoursLastYear', u'TotalWorkingYears', u'PerformanceRating', u'MonthlyIncome', u'PercentSalaryHike']
        if(type == 'genx'):
            numerical = [u'YearsSinceLastPromotion', u'TrainingHoursLastYear', u'TotalWorkingYears', u'PerformanceRating', u'MonthlyIncome', u'PercentSalaryHike']
        if(type == 'babyboomer'):
            numerical = [u'YearsSinceLastPromotion', u'TrainingHoursLastYear', u'TotalWorkingYears', u'PerformanceRating', u'MonthlyIncome', u'PercentSalaryHike']                    
        return numerical

    def getClassifierCatagoricalColumns(self, type):
        catagorical = []
        if(type == 'millenial'):
            catagorical = [u'Gender', u'MaritalStatus', u'Education']
        if(type == 'genx'):
            catagorical = [u'Gender', u'MaritalStatus', u'Education']
        if(type == 'babyboomer'):
            catagorical = [u'Gender', u'MaritalStatus', u'Education']                        
        return catagorical        

if __name__ == '__main__':
    dataLoader = DataLoader('../../../../input/millenial.csv')
    attrition = dataLoader.getData()
    print(attrition)       