from flask import Flask, render_template, request
import numpy as np
import Iris as a

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/predict", methods=['POST'] )
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    arr1 = arr.astype(np.float)
    output = a.predict(a.classes,a.thetas,arr1)
    return render_template('result.html', data=output)

@app.route("/about")
def home1():
    return render_template('about.html')

if __name__=="__main__":
    app.run(debug=True)
