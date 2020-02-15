from flask import Flask
from flask import render_template
from datetime import time
from flask import Flask, render_template, jsonify, Markup
from stock_scraper import get_data
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import json

#D3 charts: https://github.com/d3/d3/wiki/Gallery
#Python NVD3: https://github.com/areski/python-nvd3
#Flask and D3: http://benalexkeen.com/creating-graphs-using-flask-and-d3/
#Full stack development: https://realpython.com/web-development-with-flask-fetching-data-with-requests/
#Flask: http://www.patricksoftwareblog.com/flask-tutorial/
#chart.js: https://www.chartjs.org/samples/latest/
#JavaScript: https://www.codecademy.com/learn/learn-javascript
#Python Interactive Chart: https://www.pluralsight.com/guides/creating-interactive-charts-with-python-pygal

# App config.
DEBUG = True
app = Flask(__name__)
app.debug = DEBUG
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route('/simple_chart', methods=['GET', 'POST'])
def simpleChart():
    legend = 'Monthly Data'
    labels = ["January", "February", "March", "April", "May", "June", "July", "August"]
    values = [10, 9, 8, 7, 6, 4, 7, 8]
    return render_template('chart.html', values=values, labels=labels, legend=legend)


@app.route("/line_chart", methods=['GET', 'POST'])
def line_chart():
    legend = 'Temperatures'
    temperatures = [73.7, 73.4, 73.8, 72.8, 68.7, 65.2,
                    61.8, 58.7, 58.2, 58.3, 60.5, 65.7,
                    70.2, 71.4, 71.2, 70.9, 71.3, 71.1]
    times = ['12:00PM', '12:10PM', '12:20PM', '12:30PM', '12:40PM', '12:50PM',
             '1:00PM', '1:10PM', '1:20PM', '1:30PM', '1:40PM', '1:50PM',
             '2:00PM', '2:10PM', '2:20PM', '2:30PM', '2:40PM', '2:50PM']
    return render_template('line_chart.html', values=temperatures, labels=times, legend=legend)


@app.route("/time_chart")
def time_chart():
    legend = 'Temperatures'
    temperatures = [73.7, 73.4, 73.8, 72.8, 68.7, 65.2,
                    61.8, 58.7, 58.2, 58.3, 60.5, 65.7,
                    70.2, 71.4, 71.2, 70.9, 71.3, 71.1]
    times = [time(hour=11, minute=14, second=15),
             time(hour=11, minute=14, second=30),
             time(hour=11, minute=14, second=45),
             time(hour=11, minute=15, second=0),
             time(hour=11, minute=15, second=15),
             time(hour=11, minute=15, second=30),
             time(hour=11, minute=15, second=45),
             time(hour=11, minute=16, second=0),
             time(hour=11, minute=16, second=15),
             time(hour=11, minute=16, second=30),
             time(hour=11, minute=16, second=45),
             time(hour=11, minute=17, second=0),
             time(hour=11, minute=17, second=15),
             time(hour=11, minute=17, second=30),
             time(hour=11, minute=17, second=45),
             time(hour=11, minute=18, second=0),
             time(hour=11, minute=18, second=15),
             time(hour=11, minute=18, second=30)]
    return render_template('time_chart.html', values=temperatures, labels=times, legend=legend)


@app.route("/data")
def data():
    return jsonify(get_data())


@app.route("/file_upload")
def file_upload():
    return render_template('FileUploader.html')


@app.route("/")
def index():
    df = pd.read_csv('data.csv').drop('Open', axis=1)
    chart_data = df.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("index.html", data=data)


@app.route('/corr_plot', methods=['GET'])
def corr_plot():
    corr_plot_element = get_heatmap()
    return render_template('corr_heatmap.html', div_placeholder=Markup(corr_plot_element))


def get_heatmap():
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
    return py.plot(fig, output_type='div')


@app.route('/simple_plot', methods=['GET'])
def simple_pyplot():
    my_plot_div = py.plot([go.Scatter(x=[1, 2, 3], y=[3, 1, 6])], output_type='div')
    return render_template('simple_scatter.html', div_placeholder=Markup(my_plot_div))


if __name__ == "__main__":
    app.run(host="localhost", port=5001)
    #app.run()
