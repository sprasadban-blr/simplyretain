# Employee Retention Analysis
Employee retention prediction and analysis

Prerequisite:
  * Install Python 3.6 https://www.python.org/downloads/release/python-368/ 
	* Install needed python libraries
		- python -m pip install --upgrade pip
		- pip install matplotlib
		- pip install seaborn
		- pip install plotly
		- pip install sklearn
		- pip install imblearn
		- pip install IPython
		- pip install pydot
		- pip install pillow
		- pip install networkx
		- pip install flask
		- pip install flask_restful
		- pip install xgboost
		- pip install graphviz
		- pip install dtreeviz
		- pip install community
		- pip install nxviz
		- pip install xlrd
		- pip install python-louvain
		- pip install wand
		- pip install mpld3
		- pip install -r requirements.txt
		
pip install pip install matplotlib
Step 1: Clone git
  * $SRC_DIR>git clone https://github.com/sprasadban-blr/simplyretain.git

Step 2: Compile SpringBoot MVC Application 
  * $SRC_DIR>mvn clean install

Step 3: Run SpringBoot MVC Application
  * $SRC_DIR>java -jar target/simplyretain-1.0.jar

Step 4: Run REST Application
  * $SRC_DIR>python ./com/sap/hr/retention/RetentionRecommendationApp.py

Step 5: Run UI application from browser
  * http://localhost:8080/

* Highlights:
	  ..1. Techniques to isolate in generic way numerical and categorical features (Feature engineering)
	  ..2. Performing statistical analysis to know what are the fields correlated to attrition
	  ..3. Each employee where they stand in company corpus (Ex: Income Vs Salary hike, Rating Vs Income)
	  ..4. Corpus level decision tree in knowing path for an employee attrition
	  ..5. Random employee attrition path based on above decision tree
	  ..6. Providing recommendation for an employee to retain based on market research report (This can be automated by doing web crawling and   NLP techniques)
