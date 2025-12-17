from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os

# import RAG & multimodal functions
from src.rag_engine import retrieve_chunks, generate_answer_local, rag_query
from src.image_to_text import image_to_answer
from src.t2i import generate_image

app = Flask(__name__)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg"}

def allowed_file(fname):
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# --------------------- Home page ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

# --------------------- Text → Answer (RAG) -----------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    res = rag_query(question)
    # res is a dict: {"question": ..., "answer": ..., "retrieved": [...]}
    return jsonify({
        "question": res.get("question", question),
        "answer": res.get("answer"),
        "retrieved": res.get("retrieved", [])
    })



# --------------------- Image → Text → Answer ----------------
@app.route("/upload-image", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    fname = secure_filename(f.filename)
    saved_path = os.path.join(UPLOAD_DIR, fname)
    f.save(saved_path)

    # Full multimodal pipeline
    result = image_to_answer(saved_path)

    return jsonify({
        "caption": result["caption"],
        "answer": result["answer"],
        "chunks": [{
            "id": c["id"],
            "title": c["title"],
            "text": c["text"][:300] + "..."
        } for c in result["fused_chunks"]],
        "image_path": saved_path
    })

# --------------------- Text → Image -------------------------
@app.route("/gen-image", methods=["POST"])
def gen_image_api():
    data = request.json
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    out_path = generate_image(prompt)
    return send_file(out_path, mimetype="image/png")

# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
