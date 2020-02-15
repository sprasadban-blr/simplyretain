from DataLoader import DataLoader

class RetentionRecommender(object):

    def __init__(self, typeUserPath, reasonFile):
        self.userPathLoader = DataLoader(typeUserPath)
        self.userPath = self.userPathLoader.getData()
        self.reasonLoader = DataLoader(reasonFile)
        self.reason = self.reasonLoader.getData()
        self.reasonRecommendationMap = {}
        self.userReasonMap = {}
        self.nodeIds = {}        
        self.constructReasonRecommendationMap('Reason to manager', 'Reason  in exit interview', 'Recommendation ')
        self.constructSimilarEmployees('UserId', 'Nodes')
        self.constructUserReasonMap('UserId', 'Names', 'Path', 'Values', 'Nodes', 'ReasonToManager', 'ReasonInExitInterview')
        
    def constructReasonRecommendationMap(self, rmKey, rxKey, recKey):
        index = 0
        for key in self.reason[rmKey]:
            managerReason = key
            if(str(managerReason) == 'nan'):
                managerReason = ""
            exitReason = self.reason[rxKey][index]                
            if(str(exitReason) == 'nan'):
                exitReason = ""
            recommendReason = self.reason[recKey][index]                
            if(str(exitReason) == 'nan'):
                exitReason = ""     

            myKey = managerReason + exitReason               
            self.reasonRecommendationMap[myKey] = recommendReason
            print("Key=%s, value=%s" %(myKey, self.reasonRecommendationMap[myKey]))
            index = index + 1

    def constructUserReasonMap(self, userKey, names, decisionPath, pathValues, nodes, rmKey, rxKey):
        index = 0
        for userId in self.userPath[userKey]:
            managerReason = self.userPath[rmKey][index]
            if(str(managerReason) == 'nan'):
                managerReason = ""
            exitReason = self.userPath[rxKey][index]                
            if(str(exitReason) == 'nan'):
                exitReason = ""

            myValue = managerReason + exitReason
            nodeValues = self.userPath[nodes][index]
            self.userReasonMap[userId] = (myValue, self.userPath[decisionPath][index], self.userPath[names][index], self.userPath[pathValues][index], nodeValues, self.nodeIds[nodeValues], managerReason, exitReason)
            index = index + 1            
    
    def constructSimilarEmployees(self, userKey, nodesKey):
        index = 0        
        for userId in self.userPath[userKey]:
            nodesValues = self.userPath[nodesKey][index]
            if(nodesValues not in self.nodeIds):
                self.nodeIds[nodesValues] = str(userId)
            else:
                userIds = self.nodeIds[nodesValues]
                userIds = str(userIds) + "," + str(userId)
                self.nodeIds[nodesValues] = userIds
            index = index + 1            
        print(self.nodeIds)
            
    def getRecommendation(self, userId):
        pathValues = ""
        decisionPath = ""
        keyValue = ""
        nodes = ""
        similarEmployees = ""
        names = ""
        managerReason = ""
        exitReason = ""        
        if(userId in self.userReasonMap):
            tupleValues = self.userReasonMap[userId]
            keyValue = tupleValues[0]
            decisionPath = tupleValues[1]
            names = tupleValues[2]            
            pathValues = tupleValues[3]
            nodes = tupleValues[4]
            similarEmployees = tupleValues[5]
            managerReason = tupleValues[6]
            exitReason = tupleValues[7]
            
        recValue = ""
        if(keyValue in self.reasonRecommendationMap):
            recValue = self.reasonRecommendationMap[keyValue]
        
        return (userId, decisionPath, names, pathValues, nodes, recValue, similarEmployees, managerReason, exitReason)
        
if __name__ == '__main__':
    retentionRecommend = RetentionRecommender('../../../../output/millenial_user_path.csv', '../../../../input/millenial_reasons.csv')
    print("796=" + str(retentionRecommend.getRecommendation(796)))
    print("127=" + str(retentionRecommend.getRecommendation(127)))
    print("585=" + str(retentionRecommend.getRecommendation(585)))
    print("999=" + str(retentionRecommend.getRecommendation(999)))    