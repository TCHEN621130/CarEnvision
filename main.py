from flask import Flask, render_template, url_for, request
import requests
from textblob import TextBlob

#ML Packages
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
  return render_template("about.html")


@app.route("/",methods=['POST'])
def predict():

    inputData = [
    #2011 2012  2013  2014  2015
    [0.27, 0.2, 0.26, 0.33, 0.21], #2016 Toyota
    [0.2, 0.26, 0.33, 0.21, 0.3], #2017
    [0.26, 0.33, 0.21, 0.3, 0.35], #2018
    [0.33, 0.21, 0.3, 0.35, 0.25], #2019
    [0.21, 0.3, 0.35, 0.25, 0.33], #2020

    [0.34, 0.33, 0.33, 0.35, 0.42], #2016 Lexus
    [0.33, 0.33, 0.35, 0.42, 0.31], #2017
    [0.33, 0.35, 0.42, 0.31, 0.36], #2018
    [0.35, 0.42, 0.31, 0.36, 0.27], #2019
    [0.42, 0.31, 0.36, 0.27, 0.11], #2020

    [0.29, 0.38, 0.37, 0.44, 0.44], #2016 Mercedes 
    [0.38, 0.37, 0.44, 0.44, 0.38], #2017
    [0.37, 0.44, 0.44, 0.38, 0.36], #2018
    [0.44, 0.44, 0.38, 0.36, 0.32], #2019
    [0.44, 0.38, 0.36, 0.32, 0.33], #2020
    
    [0.38, 0.41, 0.28, 0.39, 0.38], #2016 AUDI
    [0.41, 0.28, 0.39, 0.38, 0.37], #2017
    [0.28, 0.39, 0.38, 0.37, 0.36], #2018
    [0.39, 0.38, 0.37, 0.36, 0.42], #2019
    [0.38, 0.37, 0.36, 0.42, 0.49], #2020
    ]

    #value = currentPrice/releasePrice
    outputData = [48, 47, 56, 85, 101, 67, 61, 71, 89, 101, 44, 53, 73, 75, 100, 34, 42, 60, 73, 101]

    mlModel = make_pipeline(PolynomialFeatures(3), Ridge())
    mlModel.fit(inputData, outputData)


    if request.method == 'POST':
      brand = request.form['brand'].lower()
      model = request.form['model'].lower()
      year = int(request.form['year'])
      

      prediction = []
      num = 6

      #This loop will get every sentiment percentage from each year
      for i in range(1, 6):
          num = num - 1
          url = "https://www.cars.com/research/" + str(brand) + "-" + str(model) + "-" + str(year - i) + "/consumer-reviews/"

          print(url)
          page = requests.get(url)
          message = str(page.content)
          score = 0
          count = 0
          begin = ""
          end = ""
          while 'card-text">' in message and '</p>' in message:
              index1 = message.index('card-text">') + 11
              index2 = message.index('</p>') - 2
              s = message[index1:index2]
              if s != "":
                  text = TextBlob(s)
                  print(s)
                  print()
                  score += text.sentiment.polarity

                  count += 1
                  print(count)
              message = message[index2 + 4:]
          avgSentiment = score/count
          prediction.append(avgSentiment)
      
      finalPredict = mlModel.predict([prediction])
      my_prediction = str(finalPredict[0])[0:5] + "% of total retail price in " + str(year + 1) + "."

    return render_template('results.html',prediction = my_prediction, brand = brand, model = model, year = year)


app.run(host="0.0.0.0")
