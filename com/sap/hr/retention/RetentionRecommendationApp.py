import flask
from flask import Flask
from flask_restful import Resource, Api
import json
from DataGeneration import DataGeneration
from RetentionRecommender import RetentionRecommender
from GenerateEmployeeGraph import GenerateEmployeeGraph

class ProcessImage(Resource):
    def __init__(self):
        ''' '''
    def post(self):
        characteristic = flask.request.get_json()
        result = ""
        #characteristic = json.jsonify(str(characteristic[2:-1])) 
        print(characteristic)
        if characteristic['salaryHike'] != '' and characteristic['monthlyIncome'] != '':
            salaryHike = characteristic['salaryHike']
            monthlyIncome = characteristic['monthlyIncome'];
            print(salaryHike)
            print(monthlyIncome);
            dataGeneration = DataGeneration("PercentSalaryHike", "MonthlyIncome", salaryHike, monthlyIncome,'result1')		
            data = str(dataGeneration.getGraph())
            data = data[2:-1]
            result = data
        if characteristic['perf_rating'] != '' and characteristic['monthlyIncome'] != '':
            perfRating = characteristic['perf_rating']
            monthlyIncome = characteristic['monthlyIncome'];
            print(perfRating)
            print(monthlyIncome);	
            dataGeneration = DataGeneration("MonthlyIncome", "PerformanceRating", monthlyIncome, perfRating,'result2')
            data = str(dataGeneration.getGraph())
            data = data[2:-1]
            result += "@@@" + data
        return str(result), 201

class ConstructDecisionTree(Resource):
    def __init__(self):
        ''' '''
    def post(self):
        characteristic = flask.request.get_json()
        #characteristic = json.jsonify(str(characteristic[2:-1])) 
        #employeeGraph = GenerateEmployeeGraph.GenerateEmployeeGraph()
        print(characteristic)
        drawGraph = GenerateEmployeeGraph()
        data = str(drawGraph.saveEmployeeGraph(characteristic, "./employeedatabase/"))
        data = data[2:-1]
        return str(data), 201

class RetentionRecommenderAPI(Resource):
    def __init__(self):
        ''' '''
        
    def get(self, type, userId):
        self.recommender = RetentionRecommender('./output/'+ type + '_user_path.csv', './input/' + type + '_reasons.csv')        
        result = {}        
        data = {}
        dataTupple = self.recommender.getRecommendation(int(userId))
        data['userid'] = dataTupple[0]
        data['decisionPath'] = dataTupple[1]
        data['names'] = dataTupple[2]        
        data['pathValues'] = dataTupple[3]        
        data['nodes'] = dataTupple[4]        
        data['recommendation'] = dataTupple[5]
        data['similaremployees'] = dataTupple[6]
        data['ReasonToManager'] = dataTupple[7]
        data['ReasonInExitInterview'] = dataTupple[8]
        
        result['Result'] = data
        return json.loads(json.dumps(result))        
    
class RetentionRecommendationService(object):
    def __init__(self):
        self.indexdata = ""
        self.app = Flask(__name__)
        self.api = Api(self.app)
        self.api.add_resource(RetentionRecommenderAPI, '/retention/<type>/<userId>')
        self.api.add_resource(ProcessImage, '/get_image')
        self.api.add_resource(ConstructDecisionTree, '/get_tree')
        self.startRESTService()
        
    def startRESTService(self):
        #self.app.run(host="10.52.33.126", port=5000)
        self.app.run(port=5000)
            
if __name__ == '__main__':
    api = RetentionRecommendationService()
