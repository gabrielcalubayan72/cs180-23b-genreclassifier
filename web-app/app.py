from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class
app = Flask(__name__)

#Load the trained model and vectorizer
model = pickle.load(open('web-app\\model.pkl', 'rb'))
vectorizer = pickle.load(open('web-app\\vectorizer.pkl', 'rb'))

#Define the route to be home
@app.route('/')
def home():
    return render_template('index.html', prediction=". . .")

#Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():
    lyrics = request.form['lyrics']
    if (lyrics != ""):
        data = [lyrics]
        vect = vectorizer.transform(data).toarray()
        predict = model.predict(vect)

        if (predict == "country"): prediction = "Country"
        if (predict == "misc"): prediction = "Misc."
        if (predict == "pop"): prediction = "Pop"
        if (predict == "rap"): prediction = "Rap"
        if (predict == "rb"): prediction = "R&B"
        if (predict == "rock"): prediction = "Rock"

        return render_template('index.html', lyrics=lyrics, prediction=prediction)
    return home()

if __name__ == "__main__":
    app.run()