from make_prediction import make_prediction, data_merge, preprocess_headlines, preprocess_posts, classify_news, calc_change_sentiment, get_news,get_stock,get_tweets
import flask
from flask import request
from markupsafe import escape
from flask import render_template,Flask, redirect, url_for, request

app = flask.Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def read_me():
    return "Final project for LighthouseLabs Data Science Bootcamp. The goal of the project is to use historical stock data in conjunction with sentiment analysis of news headlines and Twitter posts, to predict the future price of a stock of interest. The headlines were obtained by scraping the website, FinViz, while tweets were taken using Tweepy. Both were analyzed using the Vader Sentiment Analyzer."

@app.route('/search',methods = ['POST','GET'])
def login():
    if(request.args):
        company, prediction = make_prediction(request.args['ticker'])
        return flask.render_template('stock.html',
                                     company=company.upper(),
                                     prediction=prediction)
    else:       
        return flask.render_template('stock.html')

if __name__=="__main__":
    # For local development:
    app.run(debug=True)
    # For public web serving:
    #app.run(host='0.0.0.0')
    app.run()







# if __name__ == '__main__':
#     from pprint import pprint
#     pprint("Checking to see what empty string predicts")
#     pprint('input string is ')
#     ticker = 'wmt'
#     pprint(ticker)
# x_input, probs = predict_price(ticker)
# pprint(f'Input values: {x_input}')
# pprint(probs)