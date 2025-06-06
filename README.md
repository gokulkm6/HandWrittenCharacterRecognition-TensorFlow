
# 🖋️ HandWritten Character Recognition using TensorFlow

A deep learning-based handwritten character classification system built using TensorFlow and Keras. This project focuses on recognizing English alphabet characters from image data, leveraging convolutional neural networks (CNNs) for high-performance pattern learning and generalization.

---

## 📌 Table of Contents

- [Project Motivation](#project-motivation)
- [Dataset Details](#dataset-details)
- [System Architecture](#system-architecture)
- [Model Architecture](#model-architecture)
- [Technical Stack](#technical-stack)
- [Training & Evaluation](#training--evaluation)
- [Usage Instructions](#usage-instructions)
- [Results](#results)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 🎯 Project Motivation

Handwritten character recognition remains a foundational challenge in pattern recognition and computer vision. From digitizing old manuscripts to enabling offline handwriting input in devices, automating this task saves time, improves accessibility, and reduces manual errors. The goal of this project is to build a scalable, high-accuracy character recognition model trained on labeled image data, serving as a base model for real-world OCR systems.

---

## 📂 Dataset Details

The model uses an English handwritten character dataset, referenced via the `english.csv` file. This CSV maps each image file to a corresponding character label.

- **Labels**: English alphabets (lowercase), e.g., `'a'` to `'z'`
- **Input format**: RGB images, preprocessed to fixed dimensions
- **Normalization**: Pixel values scaled to `[0,1]`

> Note: All image paths in `english.csv` must be valid relative paths in the repository.

---

## 🧱 System Architecture

```text
+-------------------+      +----------------+      +---------------------+
| Raw Character PNG | ---> | Preprocessing  | ---> | CNN Classification  |
+-------------------+      | - Resize (128) |      | - Conv Layers       |
                           | - Normalize    |      | - Dense & Softmax   |
                           +----------------+      +---------------------+
```

---

## 🏗️ Model Architecture

The network consists of three convolutional blocks followed by dense layers:

```python
Sequential([
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')  # 26 classes (a-z)
])
```

---

## ⚙️ Technical Stack

| Component      | Description                              |
|----------------|------------------------------------------|
| Language       | Python 3.8+                              |
| Framework      | TensorFlow + Keras                       |
| Data Handling  | Pandas, NumPy                            |
| Image Tools    | OpenCV, Pillow (PIL)                     |
| Preprocessing  | Resize → Normalize → Encode Labels       |
| Model Type     | CNN with Dropout regularization          |
| Evaluation     | Accuracy, Loss (CrossEntropy)            |

---

## 🏁 Training & Evaluation

| Parameter             | Value         |
|----------------------|---------------|
| Epochs               | 15            |
| Batch Size           | 32            |
| Optimizer            | Adam          |
| Loss Function        | Sparse Categorical Crossentropy |
| Validation Split     | 20%           |
| Input Image Size     | 128x128 (resize) |
| Model Input Shape    | 224x224x3 (note inconsistency)   |

> **Warning:** Preprocessing resizes to 128x128, but the model expects 224x224 input. Adjust one of these for consistency.

---

## 🚀 Usage Instructions

### 🔧 Setup

```bash
git clone https://github.com/gokulkm6/HandWrittenCharacterRecognition-TensorFlow.git
cd HandWrittenCharacterRecognition-TensorFlow

pip install -r requirements.txt  # or install individually:
pip install tensorflow pandas numpy opencv-python pillow scikit-learn
```

### ▶️ Run the Training

Launch the notebook or script:

```bash
jupyter notebook hcr.ipynb
```

Or run as a Python script after converting it from notebook.

### 🔍 Test on Custom Image

Save your handwritten image as `a.png` and run the following:

```python
test_image = load_and_preprocess_image('a.png')
test_image = np.expand_dims(test_image, axis=0)
pred = model.predict(test_image)
decoded = label_encoder.inverse_transform(np.argmax(pred, axis=1))
print(f"Predicted character: {decoded[0]}")
```

---

## 📈 Results

- **Training Accuracy**: ~85.53%
- **Validation Accuracy**: ~69.35%
- **Final Test Prediction**: `'k'` for `a.png`

> Model learns to generalize well on a modest dataset; further gains require better regularization and data augmentation.

---

## ⚠️ Limitations

- Resizing inconsistency between preprocessing (`128x128`) and model input (`224x224`)
- No data augmentation or advanced regularization strategies
- Model assumes well-aligned, clean character input; may fail with noisy or cursive handwriting

---

## 🔮 Future Enhancements

- Integrate real-time handwriting input using OpenCV GUI
- Add data augmentation (rotation, shift, noise)
- Fine-tune with transfer learning (e.g., MobileNetV2)
- Save/Load model (`model.save()` / `load_model`)
- Deploy as a Flask or FastAPI web service
- Add support for digits and uppercase characters

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

> Developed and maintained by [@gokulkm6](https://github.com/gokulkm6)
