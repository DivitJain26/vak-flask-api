from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)

# Load Model
model = joblib.load("vak_svm_model.pkl")
tfidf = joblib.load("vak_tfidf.pkl")


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/check_comment", methods=["POST"])
def predict():
    data = request.get_json()
    comment = data.get("comment", "")

    cleaned = preprocess(comment)
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)[0]

    response = {
        "comment": comment,
        "result": "Abusive" if pred == 1 else "Safe",
        "flag": bool(pred),
    }
    
    print(response)

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
