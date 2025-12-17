# Monument Storyteller RAG 🏛️✨

**Monument Storyteller** is a local, privacy-focused Multimodal Retrieval-Augmented Generation (RAG) system designed to bring historical monuments to life. It combines natural language processing, computer vision, and generative AI to answer questions, analyze uploaded images of monuments, and generate artistic visualizations of heritage sites.

![Dashboard UI](assets/ui_dashboard.jpeg)

---

## 🚀 Features

This project implements three core capabilities powered by local AI models:

### 1. 📖 Context-Aware Q&A (Text-to-Text)
Ask questions about monuments (e.g., *"Who built the Taj Mahal?"*). The system uses **RAG** to retrieve relevant facts from a Wikipedia-sourced vector index (FAISS) and generates accurate answers using a local LLM (FLAN-T5).

![Text to Answer Demo](assets/demo_qa.jpeg)

### 2. 👁️ Visual Knowledge Retrieval (Image-to-Text-to-Answer)
Upload an image of a monument. The system uses **BLIP** for captioning and **CLIP** for visual embedding retrieval to identify the monument and provide a detailed explanation or answer based on the visual content.

![Image to Answer Demo](assets/demo_image_analysis.jpeg)

### 3. 🎨 Creative Visualization (Text-to-Image)
Describe a scene (e.g., *"Taj Mahal with moon"*), and the system utilizes **Stable Diffusion v1.5** to generate high-quality artistic renditions locally.

![Text to Image Demo](assets/demo_generation.jpeg)

---

## 🏗️ Architecture

The system follows a modular architecture separating the frontend (Flask) from the inference engines (RAG, CLIP, Diffusers).

![System Architecture](assets/architecture.png)

---

## 🛠️ Tech Stack

* **Backend Framework:** Flask (Python)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** * Text: `all-MiniLM-L6-v2`
    * Vision: `clip-ViT-B-32`
* **AI Models (Local):**
    * **Q&A:** `google/flan-t5-small`
    * **Captioning:** `Salesforce/blip-image-captioning-large`
    * **Image Generation:** `runwayml/stable-diffusion-v1-5`
* **Data Processing:** Pandas, NumPy, PyTorch

---

## 📂 Directory Structure

```plaintext
monument-storyteller-rag/
├── app.py                   # Main Flask application entry point
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
├── assets/                  # Screenshots and Architecture diagrams
├── src/                     # Source code for RAG, Captioning, and T2I modules
│   ├── rag_engine.py
│   ├── image_to_text.py
│   ├── clip_embed.py
│   └── t2i.py
├── notebooks/               # Jupyter notebooks for data pipeline
│   ├── 01-data-ingest.ipynb
│   ├── 02-preprocess-chunk.ipynb
│   ├── 03-embeddings-index.ipynb
│   └── 04-clip-index.ipynb
├── templates/               # HTML templates for the web interface
│   └── index.html
└── Data/                    # Storage for raw text, processed chunks, and indices
    ├── raw/
    ├── processed/
    └── embeddings/
```
## ⚙️ Installation & Setup
### 1. Clone the Repository
git clone [https://github.com/manishh18/monument-storyteller-rag.git](https://github.com/manishh18/monument-storyteller-rag.git)
cd monument-storyteller-rag

