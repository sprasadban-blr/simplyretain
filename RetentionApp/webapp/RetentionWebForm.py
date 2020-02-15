''' pip install wtforms '''
''' pip install Flask-Session '''
#from RetentionApp.webapp.DataPreProcessing import DataPreProcessing
from DataPreProcessing import DataPreProcessing
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
''' pip install flask '''
''' Flask Examples: https://pythonspot.com/flask-web-app-with-python/'''
''' https://stackoverflow.com/questions/45227076/how-to-send-data-from-flask-to-html-template '''
''' DT visualization: https://explained.ai/decision-tree-viz/ '''
''' https://pythonhosted.org/Flask-Session/ '''
from flask import Flask, render_template, flash, request, redirect, url_for, session, Markup
#from flask.ext.session import Session
from wtforms import Form, TextField, validators
from sklearn import tree
from dtreeviz.trees import *
from wand.api import library
import wand.color
import wand.image
import pandas as pd
import base64
import plotly.graph_objs as go
import plotly.offline as py
import os
 
# App config.
DEBUG = True
app = Flask(__name__)
app.debug = DEBUG
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
#SESSION_TYPE = 'filesystem'
#Session(app)
 
class HRRetentionWebForm(Form):
    username = TextField('username', validators=[validators.required()])
    password = TextField('password', validators=[validators.required()])
    
    def saveImage(self, svgBytes, fileName):
        with wand.image.Image() as image:
            with wand.color.Color('transparent') as background_color:
                library.MagickSetBackgroundColor(image.wand, background_color.resource) 
            image.read(blob=svgBytes)
            png_image = image.make_blob("png32")
    
        with open(fileName, "wb") as out:
            out.write(png_image)
            out.close()
    
    def getClassifierNumericalColumns(self, ageCat):
        numerical = []
        if(ageCat == 'millenial'):
            numerical = [u'YearsSinceLastPromotion', u'TrainingHoursLastYear', u'TotalWorkingYears', u'PerformanceRating', u'MonthlyIncome', u'PercentSalaryHike']
        if(ageCat == 'genx'):
            numerical = [u'YearsSinceLastPromotion', u'TrainingHoursLastYear', u'TotalWorkingYears', u'PerformanceRating', u'MonthlyIncome', u'PercentSalaryHike']
        if(ageCat == 'babyboomer'):
            numerical = [u'YearsSinceLastPromotion', u'TrainingHoursLastYear', u'TotalWorkingYears', u'PerformanceRating', u'MonthlyIncome', u'PercentSalaryHike']                    
        return numerical
    
    def getClassifierCatagoricalColumns(self, ageCat):
        catagorical = []
        if(ageCat == 'millenial'):
            catagorical = [u'Gender', u'MaritalStatus', u'Education']
        if(ageCat == 'genx'):
            catagorical = [u'Gender', u'MaritalStatus', u'Education']
        if(ageCat == 'babyboomer'):
            catagorical = [u'Gender', u'MaritalStatus', u'Education']                        
        return catagorical        
     
    def displayImpFeatures(self, features, columns):
        index = 0
        print("Important features:\n")
        for feature in features:
            print(columns[index] + "=" + str(feature))
            index = index + 1
        ''' Another way of getting important features '''
        print(dict(zip(columns, features)))
        
    def get_heatmap(self, attrition_data):
        attrition_data = pd.read_csv('../../input/emp_data.csv')
        cols = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'TotalWorkingYears',
                'TrainingTimesLastYear', 'WorkLifeBalance',
                'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                'YearsWithCurrManager']
        corr_data = [
            go.Heatmap(
                z=attrition_data[cols].astype(float).corr().values,
                x=attrition_data[cols].columns.values,
                y=attrition_data[cols].columns.values,
                colorscale='Viridis',
                reversescale=False,
                opacity=1.0
    
            )
        ]
        layout = go.Layout(
            title='Pearson Correlation of numerical features',
            xaxis=dict(ticks='', nticks=36),
            yaxis=dict(ticks=''),
            width=900, height=700,
    
        )
        fig = go.Figure(data=corr_data, layout=layout)
        diagram = py.plot(fig, output_type='div')
        return diagram
        
    def generateEmpGraph(self, attrition_data, field_value1, field_value2):
        attrition_data = pd.read_csv('../../input/millenial.csv')
        fig = plt.figure()   
        attr_yes = attrition_data[attrition_data["Attrition"] == "Yes"][["PercentSalaryHike", "MonthlyIncome"]]
        attr_no = attrition_data[attrition_data["Attrition"] == "No"][["PercentSalaryHike", "MonthlyIncome"]]
        max_x = max(len(attr_yes), len(attr_no))+1
         
        plt.xticks(np.arange(0, max_x, 5.0))
        print("########",attr_yes)
        plt.plot(attr_yes["PercentSalaryHike"], attr_yes["MonthlyIncome"],   '.', color='red', label= 'Attrition = Yes')
        plt.plot(attr_no["PercentSalaryHike"], attr_no["MonthlyIncome"],  '.', color='cyan', label='Attrition = No')
        plt.title("MonthlyIncome" + " vs. " + "PercentSalaryHike")
        plt.xlabel("PercentSalaryHike")
        plt.ylabel("MonthlyIncome")
        plt.legend(loc=1)
        plt.plot(int(field_value1), int(field_value2), 'o', label = 'Employee Data')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path.replace('\\','/')        
        print(dir_path) 
        fig.savefig(dir_path+"/static/resources/employeeData.png")
        plt.clf()

@app.route('/sap/loadfile', methods=['GET', 'POST'])
def loadfile():
    username = session.get('username')    
    form = HRRetentionWebForm(request.form)    
    if request.method == 'POST':
        fileContent = request.form['myTextarea']
        print(type(fileContent))
        
        file = open('dtanalysis.csv', 'w')
        file.write(fileContent)
        file.close()
        #retention = pd.read_csv('dtanalysis.csv')
        config_path = "./config/config"
        data = pd.read_csv("./dtanalysis.csv")
        data_preprocessed = DataPreProcessing(data, config_path)
        retention = data_preprocessed.get_pre_processed_data()
        
        '''
        li = list(fileContent.splitlines(False))
        # Get columns list
        cols = {}
        index = 0;
        colsList = li[0].split(",")
        for col in colsList:
            cols[index] = col
            index = index + 1
            
        # extract data other than columns
        li = li[1:]
        print(li[1])
        print(li[2])
        # Convert to pandas data frame
        retention = pd.DataFrame([sub.split(",") for sub in li])
        # Replace columns names
        retention.rename(columns=cols, inplace=True)
        print(retention.columns)
        print(retention)
        '''
        
        #numerical = form.getClassifierNumericalColumns('millenial')
        numerical = data_preprocessed.get_numerical()
        #categorical = form.getClassifierCatagoricalColumns('millenial')
        categorical = data_preprocessed.get_categorical()
        categorical.remove('Attrition')
        # Store the categorical data in a dataframe called attrition_cat
        attrition_cat = retention[categorical]
        
        #One hot vector for categorical columns
        attrition_cat = pd.get_dummies(attrition_cat)
        #print(attrition_cat.head(3))
        
        # Store the numerical features to a dataframe attrition_num
        attrition_num = retention[numerical]
        # Concat the two dataframes together columnwise
        X = pd.concat([attrition_num, attrition_cat], axis=1)
        # Define a dictionary for the target mapping
        print(retention["Attrition"].head(5))
        # Use the pandas apply method to numerically encode our attrition target variable
        #target_map = {'Yes':1, 'No':0}        
        #y = retention["Attrition"].apply(lambda x: target_map[x])
        y = retention["Attrition"].apply(lambda x: int(x))
        
        regr = tree.DecisionTreeClassifier(max_depth=4)
        regr.fit(X, y)
        form.displayImpFeatures(regr.feature_importances_, retention.columns)
        print(X.columns.values)
        
        '''viz = dtreeviz(regr,
                       X,
                       y,
                       target_name='Attrition',
                       feature_names = X.columns.values,
                       class_names=["yes", "no"])'''
        viz = dtreeviz(regr,
                       X,
                       y,
                       target_name='Attrition',
                       feature_names = X.columns.values,
                       class_names=[1, 0])        
        
        svg_code = viz.svg()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path.replace('\\','/')
        print(dir_path)
        form.saveImage(svg_code.encode('utf8'), dir_path+'/static/resources/dt3.png')
        #pic_data = open("C:/Users/I050385/github/HR/SimplyRetain/RetentionAnalysis/RetentionApp/webapp/dt3.png", 'rb').read()
        #print(pic_data)
        #session['dtchart'] = str(base64.b64encode(pic_data))
        # Load charts
        #session['retention'] = retention.to_json() # Dataframe to JSON payload
        
        regr = tree.DecisionTreeClassifier(max_depth=4)
        regr.fit(X, y)
        #Random Single observation for prediction.
        employeeId = np.random.randint(0, len(X))        
        singleX = X.iloc[employeeId,::]  # random sample from training
        print(singleX)
        
        viz = dtreeviz(regr,
                       X,
                       y,
                       target_name='Attrition',                    
                       feature_names=X.columns.values,
                       class_names=[1, 0],
                       X=singleX) # need to give single observation for prediction
        
        svg_code = viz.svg()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path.replace('\\','/')
        print(dir_path)
        form.saveImage(svg_code.encode('utf8'), dir_path+'/static/resources/dt4.png')
        
        return redirect(url_for('displaycharts', retention=retention, employeeId=employeeId))
        #return redirect(url_for('displaycharts'))        
        #return render_template("DisplayCharts.html", dtchart=str(base64.b64encode(pic_data)))
    return render_template("FilePicker.html")

@app.route("/sap/displaycharts", methods=['GET', 'POST'])
def displaycharts():
    '''
    dataImage = request.get('dtchart')
    print(dataImage)
    data = "data:image/png;base64" + str(dataImage)
    '''
    #retention = request.args['retention']  # counterpart for url_for()
    #retention = pd.read_json(session['retention'])       # From JSON to Dataframe
    retention = request.values.get('retention') # counterpart when not sure session Or request
    employeeId = request.values.get('employeeId')
    form = HRRetentionWebForm(request.form)    
    corr_plot_element = form.get_heatmap(retention)
    #form.generateEmpGraph(retention, 35000, 10)    
    return render_template("DisplayCharts.html", employeeId=employeeId,
                           #employeePlot="../static/resources/employeeData.png",                            
                           decision_tree3="../static/resources/dt3.png", 
                           decision_tree4="../static/resources/dt4.png", 
                           div_placeholder=Markup(corr_plot_element))
    '''return render_template("DisplayCharts.html", decision_tree="../static/resources/dt3.png")'''


@app.route("/sap", methods=['GET', 'POST'])
def logon():
    ''' reset the session data '''
    session.clear()
    form = HRRetentionWebForm(request.form)
    print(form.errors)
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']        
        if form.validate():
            flash('Success!')
            if username.lower() == 'admin' and password == 'admin':
                session['username'] = username
                return redirect(url_for('loadfile'))
            else:
                flash('Error: Either user name or password is invalid.')
        else:
            flash('Error: Please enter log on credentials.')
    return render_template('index.html', form=form)

@app.before_request
def session_management():
    ''' make the session last indefinitely until it is cleared '''
    session.permanent = True
    
if __name__ == "__main__":
    app.run(host="localhost", port=5002)
    #app.run()