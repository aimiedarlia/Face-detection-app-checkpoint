# 👤 Face Detection App using Viola-Jones Algorithm (Streamlit + OpenCV)

This project is a **real-time face detection application** built using **Streamlit** and **OpenCV**.  
It applies the **Viola-Jones Haar Cascade Classifier** to detect faces from either:

- A **webcam feed**
- An **uploaded image**

Users can **customize face detection** using multiple interactive features.

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| Real-time webcam detection | Detects faces live using your laptop’s camera |
| Image upload detection | Detect faces from.jpg/.jpeg/.png images |
| Adjustable detection parameters | `scaleFactor` and `minNeighbors` sliders |
| Editable rectangle color | Pick any detection color using a color picker |
| Save processed images | Save frames or images with drawn faces to your device |
| Detection instructions | Clear user guidance inside the UI |

---

## 🔧 How It Works

The app uses the Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to identify human faces.  
After detection, rectangles are drawn around each face using a user-selected color.

Detection parameters:

| Parameter | Function |
|----------|----------|
| `scaleFactor` | Controls scaling of image pyramid (affects speed + sensitivity) |
| `minNeighbors` | Determines strictness of detection (affects false positives) |

---

## 📂 Project Structure

```

📁 Face Detection App
│
├─ face_detection_app.py      → Main Streamlit application
├─ requirements.txt           → Dependencies
└─ README.md                  → Documentation (this file)

```

---

## ▶️ How to Run the App Locally

### 1️⃣ Clone or download the project
```

git clone <your-repository-url>
cd face-detection-app

```

### 2️⃣ Install dependencies
```

pip install -r requirements.txt

```

### 3️⃣ Run the Streamlit app
```

streamlit run face_detection_app.py

```

---

## 📌 Notes for Google Colab Users

Streamlit doesn't run normally on Colab unless routed via a tunnel (e.g., `pyngrok`).  
To run on Colab:
```

!pip install streamlit pyngrok opencv-python-headless

```

Then:
```

from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
!streamlit run face_detection_app.py &>/dev/null&

```

---

## 💡 Tips for Better Detection

- Ensure good lighting and face the camera directly
- If no face is detected → lower **Min Neighbors**
- If false detections occur → raise **Min Neighbors**
- If processing is slow → raise **Scale Factor**

---

## 📜 License

This project is free to use for **education and research purposes**.

---

## 👩‍💻 Author

Developed with ❤️ by **Anne Wanjiru** (with help from ChatGPT)

```

---

## 📌 requirements.txt

```text
streamlit
opencv-python
opencv-python-headless
numpy
pyngrok
```

> If you're running **locally (not in Colab)**, you can remove `opencv-python-headless` to avoid conflicts.


