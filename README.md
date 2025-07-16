# CP8318 – Visual Question Answering (VQA)

A Keras/TensorFlow implementation of deep learning models for VQA tasks. This code can:

1. Pre-process the **Easy-VQA** and **Abstract Scenes** datasets
2. Train Bag-of-Words **or** LSTM question encoders paired with a lightweight CNN image encoder
3. Track training history & metrics, export PDF plots, and evaluate precision/recall/F1
4. Re-use the provided pre-trained weights for quick demos

---

## Repository layout

* `easy-vqa/` – complete Easy-VQA pipeline (≈4 k train + 1 k test images)
* `vqa-abstract-scenes/` – experiments on the harder Abstract-Scenes dataset (20 k images, 60 k Q/A pairs)
* `clevr/` – early prototype / scratch work
* `models/`, `history/`, `plots/` – auto-generated during training

---

## Quick Start 🚀

### 1. Environment (Python 3.9+)
```bash
python -m venv venv && source venv/bin/activate   # or use Conda
pip install -r requirements.txt                   # tensorflow >=2.9, scikit-learn, scikit-image, matplotlib …
```

### 2. Easy-VQA demo
```bash
cd easy-vqa
python combine_data.py   # one-off: merges train & test splits into ./data/
python vqa.py            # trains 10 epochs, saves weights to first_model/
```
Expected accuracy: **98.9 % train / 97.5 % validation**

### 3. Abstract-Scenes experiments (GPU recommended)
```bash
cd ../vqa-abstract-scenes
python main.py           # runs a batch of predefined experiments
```
Checkpoints are written to `models/` each epoch.  To visualise learning curves:
```bash
python plots.py          # outputs PDFs to ./plots/
```

---

## TL;DR Results

Bag-of-Words models (5 conv layers, 5 epochs)
* **Yes/No** – 86 % train, 78 % test
* **Top-10 answers** – 71 % train, 69 % test
* **Top-100 answers** – 61 % train, 56 % test

Sequential (LSTM) models
* **Yes/No** – 58 % train, 56 % test
* **Top-10 answers** – 45 % train, 32 % test
* **Top-100 answers** – 34 % train, 17 % test

---

## Authors & Attribution
*Matthew Bush* · *Volodymyr Tanczak*  — course project for grad school course **CP8318**
