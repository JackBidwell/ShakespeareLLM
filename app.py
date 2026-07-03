from flask import Flask, jsonify, render_template, request

from inference import serve

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", models=serve.available_models())


@app.route("/api/models")
def api_models():
    return jsonify(serve.available_models())


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True, silent=True) or {}

    prompt = (data.get("prompt") or "").strip()
    model_name = data.get("model", "gpt2")
    temperature = float(data.get("temperature", 0.8))
    top_k = int(data.get("top_k", 40))
    max_new_tokens = max(1, min(int(data.get("max_new_tokens", 200)), 400))

    if not prompt:
        return jsonify({"error": "A prompt is required."}), 400

    try:
        output = serve.generate(
            model_name,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"output": output})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
