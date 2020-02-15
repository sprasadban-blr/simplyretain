import pandas as pd
import numpy as np
import scipy
import json
import matplotlib.pyplot as plt
import os
import base64

class DataGeneration(object):
	def __init__(self, field_name1, field_name2, field_value1, field_value2, file_name):
		self.field_name1 = field_name1
		self.field_name2 = field_name2
		self.field_value1 = field_value1
		self.field_value2 = field_value2
		self.file_name = file_name
		self.data = pd.read_excel("./employeedatabase/tempData.xlsx")
		print(self.data.head())
		
		
	def getGraph(self):
		try:
			os.remove("./employeedatabase/"+self.file_name+".png")
		except OSError:
			pass
		print( "##################", self.field_name1, self.field_name2, self.field_value1, self.field_value2, self.file_name)
		fig = plt.figure()		
		attr_yes = self.data[self.data["Attrition"] == "Yes"][[self.field_name1, self.field_name2]]
		attr_no = self.data[self.data["Attrition"] == "No"][[self.field_name1, self.field_name2]]
		max_x = max(len(attr_yes), len(attr_no))+1
		
		plt.xticks(np.arange(0, max_x, 5.0))
		print("########",attr_yes)
		plt.plot(attr_yes[self.field_name1], attr_yes[self.field_name2],   '.', color='red', label= 'Attrition = Yes')
		plt.plot(attr_no[self.field_name1], attr_no[self.field_name2],  '.', color='cyan', label='Attrition = No')
		fig.savefig("./employeedatabase/temp_result1.png")
		#x_axis = np.linspace(0,max_x,max_x+1)
		#plt.plot(x_axis,np.full(len(x_axis), field_value)) 
		plt.plot(int(self.field_value1), int(self.field_value2), 'o', label = 'Employee Data')
		plt.title(self.field_name2 + " vs. " + self.field_name1)
		plt.xlabel(self.field_name1)
		plt.ylabel(self.field_name2)
		plt.legend(loc=1)
		#plt.show()
		fig.savefig("./employeedatabase/"+self.file_name+".png")
		plt.clf()		
		#plt.close('all')
		pic_data = open("./employeedatabase/"+self.file_name+".png", 'rb').read()
		return str(base64.b64encode(pic_data))
	
if __name__ == '__main__':
    dataGen = DataGeneration()
    dataGen.getGraph()