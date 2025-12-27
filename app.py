from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/check", methods=["POST"])
def check_news():
    data = request.get_json()
    news_text = data.get("news", "")

    if not news_text.strip():
        return jsonify({"error": "Empty news text"}), 400

    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)[0]

    verdict = "FAKE" if prediction == 0 else "REAL"
    return jsonify({"verdict": verdict})

if __name__ == "__main__":
    app.run()

