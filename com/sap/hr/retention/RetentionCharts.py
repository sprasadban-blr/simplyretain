from DataLoader import DataLoader

class RetentionCharts(object):
    def __init__(self, dataFile):
        self.dataLoader = DataLoader(dataFile)
        self.retention = self.dataLoader.getData()
        
if __name__ == '__main__':
    charts = RetentionCharts('../../../../input/millenial.csv')
    charts.display()        