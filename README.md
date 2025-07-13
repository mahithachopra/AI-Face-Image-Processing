# 🎨 Multi-Face RGB + Skin Tone Analyzer

This Streamlit app processes uploaded images to detect multiple human faces and extract RGB values from specific facial regions (forehead, cheeks) to estimate skin tone. It displays annotated images, RGB previews, and a pie chart of skin tone distribution. You can also download the results in CSV, JSON, and ZIP (annotated images) formats.


## 🚀 Setup and Run Instructions

### 1. Clone the Repository
git clone https://github.com/mahithachopra/AI-Face-Image-Processing.git
cd AI-Face-Image-Processing
### 2. Install Dependencies
Ensure you have Python 3.7+ installed. Then run:
pip install -r requirements.txt
### 3. Run the Application
streamlit run app.py

## ⚙️ Technical Decisions and Implementation

Face Detection: Used OpenCV's Haar Cascade for frontal face detection for its speed and simplicity.

Color Sampling: Extracted RGB from three key facial regions—forehead, left cheek, and right cheek—for a better skin tone approximation.

Skin Tone Classification: Applied average lightness of the RGB values to categorize skin tones as Fair, Medium, or Dark.

Visualization: Used Matplotlib for pie chart rendering and HTML formatting within Streamlit for RGB color swatches.

Performance: Leveraged in-memory ZIP and Streamlit expanders to handle multiple images efficiently without cluttering the UI.

## 🧠 Challenges & Improvements
Challenges:
Ensuring pixel sampling points don’t go out of bounds for smaller faces.

Handling inconsistent lighting conditions in uploaded images, which could affect color accuracy.

Keeping the UI responsive while processing multiple images.

## Possible Improvements:
Integrate Dlib or Mediapipe for more accurate face landmark detection.

Add support for batch processing through folders or drag-and-drop.

Provide HSV and Lab color space analysis for more robust skin tone estimation.

Include face alignment and resizing for more consistent sampling.

## 📂 File Overview
app.py – Streamlit app with UI and core logic

requirements.txt – Dependency list for easy setup

📷 Sample Use Case
<img width="1906" height="846" alt="Screenshot 2025-07-13 163235" src="https://github.com/user-attachments/assets/30d0529f-0bf4-4420-9751-394983f9946b" />
<img width="1781" height="703" alt="Screenshot 2025-07-13 163249" src="https://github.com/user-attachments/assets/4453ffd4-dd3e-43e4-9c73-3a46933ea67c" />
<img width="1886" height="798" alt="Screenshot 2025-07-13 163312" src="https://github.com/user-attachments/assets/8cf0027e-32f1-4a05-98aa-a49a7dfd5bd7" />
<img width="1793" height="817" alt="Screenshot 2025-07-13 163324" src="https://github.com/user-attachments/assets/2066c357-702e-4303-a7bd-7e544c2994c1" />

