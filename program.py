from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle as pkl

app = Flask(__name__)

@app.route("/")
def Home():
    return render_template("home.html")

@app.route("/car-price-prediction")
def Project():
    df = pd.read_csv("Data.csv")
    companies = sorted(df["company"].unique())
    names = sorted(df["name"].unique())
    return render_template("car-price-prediction.html", companies = companies, names = names)

@app.route("/car-price-prediction-result")
def Result():
    company = request.args.get("company")
    name = request.args.get("name")
    year = request.args.get("year")
    kms_driven = request.args.get("kms_driven")
    fuel_type = request.args.get("fuel_type")

    pipe = pkl.load(open('CarPricePredictionModel.pkl', 'rb'))

    myinput = np.array([name, company, year, kms_driven, fuel_type]).reshape(1, 5)
    columns = ['name', 'company', 'year', 'kms_driven', 'fuel_type']
    data = pd.DataFrame(columns = columns, data = myinput)

    result = int(pipe.predict(data)[0][0])

    return render_template("car-price-prediction-result.html", company = company, name = name, year = year, kms_driven = kms_driven, fuel_type = fuel_type, result = result)
if __name__ == "__main__":
    app.run(debug = True, port = 8000)