import matplotlib.pyplot as plt
import networkx as nx
import json
import base64

class GenerateEmployeeGraph(object):
    def __init__(self):
        ''' '''
    def replaceUser(self, user, nodeValue):
        nodeText = nodeValue.replace('X', 'Employee_Id:'+str(user))
        start = nodeText.index('[')
        end = nodeText.index(']')
        return nodeText[:start] + nodeText[end+1:]

    def saveEmployeeGraph(self, employeeJson, fileLocPath):
        self.employee = employeeJson
        self.features = self.employee['Result']['names'].split("-->")
        print(self.features)
        self.nodes = self.employee['Result']['nodes'][1:-1].split(" ")
        print(self.nodes)
        self.nodeEdges = [(self.nodes[i], self.nodes[i+1]) for i in range(len(self.nodes) - 1)]
        print(self.nodeEdges)
        self.pathValues = self.employee['Result']['pathValues'] 
        nodeList = self.pathValues.split("-->")
        self.nodesMap = {}
        nodeCount = 0
        for node in nodeList:
            nodeValues = node.split(":")
            print(nodeValues)
            nodeNum = nodeValues[0].split("node ")[1]
            afterSplit = nodeNum.split(" ")
            #nodeText = "node #" + afterSplit[0]
            nodeText = self.features[nodeCount]
            if(len(nodeValues) > 1):
                print(nodeValues[1])
                nodeText = nodeText + "\n" + self.replaceUser(self.employee['Result']['userid'], nodeValues[1]) 
            else:
                nodeText = nodeText + "\n" + nodeValues[0]                 
            print(nodeText)
            self.nodesMap[afterSplit[0]] = nodeText
            nodeCount = nodeCount + 1
        print(self.nodesMap)
        
        fig = plt.figure()
        G = nx.DiGraph(directed=True)
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.nodeEdges)
        nx.draw(G, with_labels=True, arrows=True, labels=self.nodesMap, node_size=1000, width=3, arrowstyle='-|>', arrowsize=12)
        plt.axis()
        fig.savefig(fileLocPath+"emppathgraph.png")        
        #plt.show()
        pic_data = open(fileLocPath+"emppathgraph.png", 'rb').read()
        return str(base64.b64encode(pic_data))
        
if __name__ == '__main__':
    employeeJson = '{"Result": {"decisionPath": "node 0-->node 1-->node 4-->leaf node 5 (No)", "userid": 17, "recommendation": NaN, "similaremployees": "17,87,96,110,111,179,240,251,261,285,371,375,419,502,504,550,578,649,742,766,767,809,843,868,993,1009,1032,1075,1244,1254,1259,1394,1413,1505,1518,1523,1546,1611,1617,1710,1727,1733,1744,1766,1774,1775,1791,1818,1853,1883,1885,1938,2003,2058,2080,2091,2174,2178,2211,2220,2243,2271,2285,2337,2358,2430,2466,2498,2521,2538,2580,2582,2601,2646,2649,2745,2770,2782,2859,2872,2875,2908,2928,2952,2976,3022,3062,3130,3168,3281,3294,3297,3300,3330,3398,3419,3438,3476,3510,3538,3577,3612,3624,3652,3674,3706,3748,3769,3789,3808,3813,3881,3893,3894,3896,4013,4040,4052,4142,4182,4223,4245,4339,4361,4431,4437,4455,4480,4509,4545,4573,4610,4618,4676,4705,4714,4715,4722,4791,4792,4860,4908,4912,4919,4932,4946,4954,5144,5169,5305,5321,5365,5376,5403,5427,5442,5467,5492,5551,5596,5613,5640,5726,5734,5751,5753,5758,5874,5896,5903,5920,5929,5997,6011,6100,6105,6135,6172,6176,6225,6228,6253,6288,6377,6398,6421,6461,6500,6557,6564,6641,6750,6794,6795,6797,6850,6879,6883,6927,6933,6980,6985", "pathValues": "node 0: (X[17,3] (= 3) <= 3.996037006378174)-->node 1: (X[17,8] (= 1) > 0.9974051713943481)-->node 4: (X[17,6] (= 0) <= 0.9539861679077148)-->leaf node 5 (No)", "nodes": "[0 1 4 5]", "names": "PerformanceRating-->MaritalStatus_Married-->Gender_Female-->No"}}'    
    employeeGraph = GenerateEmployeeGraph()
    employeeGraph.saveEmployeeGraph(employeeJson, "./output/")