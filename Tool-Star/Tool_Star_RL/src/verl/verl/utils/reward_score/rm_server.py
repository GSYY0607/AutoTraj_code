import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "/to/your/path/model"

device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to(device)
model.eval()

@app.route("/score", methods=["POST"])
def score():
    try:
        text = request.json["trajectory"]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits[0, 0].item()

        return jsonify({"score": score})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
