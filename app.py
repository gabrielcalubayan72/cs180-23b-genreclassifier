from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class
app = Flask(__name__)

#Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

#Define the route to be home
@app.route('/')
def home():
    return render_template('index.html')

#Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():
    lyrics = request.form['lyrics']
    data = [lyrics]
    vect = vectorizer.transform(data).toarray()
    prediction = model.predict(vect)
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run()