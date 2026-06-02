# 🔬 CV Lab — CIFAR-10 Vision System
**Course:** Computer Vision | **Dept.:** Creative Technology  
**Lab Instructor:** Sania Akhtar

---

## What This Project Does

A full Streamlit web app covering your Computer Vision lab objectives:

| Module | Lab Objective |
|--------|--------------|
| 🧠 Image Classifier | CNN trained on CIFAR-10 with morphological preprocessing |
| 🔬 Morphological Ops | Erosion, Dilation, Opening, Closing, Gradient (Lab Tasks 5 & 6) |
| 🎯 Hit-or-Miss | Pattern detection with custom B1/B2 structuring elements (Lab Task 1) |
| 🌀 Image Fusion | Pixel-level & Laplacian Pyramid fusion (Lab Task 3) |

---

## Setup (Local — Windows / Mac / Linux)

### Step 1 — Clone / Download
Place all files in one folder, e.g. `cv_project/`

```
cv_project/
├── app.py
├── model.py
├── requirements.txt
└── README.md
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Pre-train the model (first run only, ~5-10 min)
```bash
python model.py
```
This downloads CIFAR-10 via Keras, trains the CNN for up to 30 epochs,
and saves `cifar10_cnn.keras` in the same folder.

> **Skip this step** if you want to train on first Streamlit use instead
> (training will start automatically when you click "Run Classification").

### Step 5 — Launch the app
```bash
streamlit run app.py
```
Opens at → http://localhost:8501

---

## Using with Kaggle Dataset

To use the **Kaggle CIFAR-10** dataset instead of the Keras built-in version:

1. Download from: https://www.kaggle.com/competitions/cifar-10
2. Extract to a folder, e.g. `data/cifar-10-batches-py/`
3. In `model.py`, replace the line:
   ```python
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
   ```
   with the Kaggle loader below:
   ```python
   import pickle, pathlib
   def load_kaggle_cifar10(data_dir="data/cifar-10-batches-py"):
       def unpickle(file):
           with open(file, 'rb') as f:
               return pickle.load(f, encoding='bytes')
       x_train, y_train = [], []
       for i in range(1, 6):
           d = unpickle(f"{data_dir}/data_batch_{i}")
           x_train.append(d[b'data'])
           y_train.extend(d[b'labels'])
       x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).transpose(0,2,3,1)
       y_train = np.array(y_train)
       test = unpickle(f"{data_dir}/test_batch")
       x_test = test[b'data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1)
       y_test = np.array(test[b'labels'])
       return (x_train, y_train), (x_test, y_test)
   (x_train, y_train), (x_test, y_test) = load_kaggle_cifar10()
   ```

---

## Running on Google Colab

```python
# Cell 1 — Install
!pip install streamlit tensorflow opencv-python-headless pyngrok -q

# Cell 2 — Upload files
from google.colab import files
uploaded = files.upload()  # upload app.py and model.py

# Cell 3 — Pre-train
!python model.py

# Cell 4 — Launch via ngrok
from pyngrok import ngrok
import subprocess, threading

def run_streamlit():
    subprocess.run(["streamlit", "run", "app.py",
                    "--server.port", "8501",
                    "--server.headless", "true"])

t = threading.Thread(target=run_streamlit)
t.start()

public_url = ngrok.connect(8501)
print(f"\n🔬 App live at: {public_url}")
```

---

## Expected CNN Accuracy

| Metric | Value |
|--------|-------|
| Training accuracy | ~88–92% |
| Test accuracy | ~80–85% |
| Training time | ~5–10 min (GPU) / ~30–60 min (CPU) |

---

## Lab Concepts Covered

- ✅ **Erosion** — removes noise, shrinks bright regions  
- ✅ **Dilation** — fills gaps, expands bright regions  
- ✅ **Opening** — erosion then dilation (removes small objects)  
- ✅ **Closing** — dilation then erosion (fills small holes)  
- ✅ **Morphological Gradient** = Dilation − Erosion (edge detection)  
- ✅ **Hit-or-Miss Transform** = Erode(A, B1) ∩ Erode(Aᶜ, B2)  
- ✅ **Image Fusion** — pixel-level & Laplacian pyramid feature-level  
- ✅ **CNN Classification** with preprocessing pipeline  
