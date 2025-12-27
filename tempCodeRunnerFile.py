from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news_text = request.form["news"]
        news_vec = vectorizer.transform([news_text])
        prediction = model.predict(news_vec)

        result = "Real News 📰" if prediction[0] == 1 else "Fake News ⚠️"
        return render_template("index.html", prediction=result, input_text=news_text)

if __name__ == "__main__":
    app.run(debug=True)
