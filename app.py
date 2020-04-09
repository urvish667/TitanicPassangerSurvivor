import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('titanic_knn.pkl', 'rb'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    pclass = int(request.form['pclass'])
    sex = int(request.form['sex'])
    embarked = int(request.form['embarked'])
    title = int(request.form['title'])
    deck = int(request.form['deck'])
    family_size = int(request.form['family_size'])
    age = int(request.form['age'])*pclass
    fare = int(request.form['fare'])/family_size
    final_feature = np.array([[pclass, sex, embarked, title, deck, family_size, age, fare]])
    survived = model.predict(final_feature)
    if survived == 1:
        return render_template('home.html', result = "Passanger is lucky!! :)")
    else:
        return render_template('home.html', result = "Sorry for Passanger!! :(")


if __name__ == "__main__":
    app.run(debug=True)