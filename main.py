from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form['message']
        clf_svm = pickle.load(open('clf.pkl','rb'))
        tfidf = pickle.load(open('tfidf.pkl','rb'))
        text = tfidf.transform([message])
        result = clf_svm.predict(text)
        message = " "
        if(result == 1):
            message = 'The text is likely wirtten by AI'
        else:
            message = "The text is likely written by Human"
    return render_template('main.html',params = message)

if __name__ == '__main__':
    app.run(debug=True)
