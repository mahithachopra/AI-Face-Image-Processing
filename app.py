import streamlit as st
import cv2
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from zipfile import ZipFile

# ----------- FACE PROCESSING FUNCTION -----------
def extract_faces_from_image(image, image_name, show_labels=True, show_keypoints=True):
    result = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for idx, (x, y, w, h) in enumerate(faces):
        face_id = idx + 1
        face_row = {
            "Image": image_name,
            "Face ID": face_id,
            "Forehead RGB": None,
            "Left Cheek RGB": None,
            "Right Cheek RGB": None,
            "Skin Tone": None
        }

        sampled_colors = []
        points = {
            "Forehead RGB": (x + w // 2, y + h // 6),
            "Left Cheek RGB": (x + w // 4, y + h // 2),
            "Right Cheek RGB": (x + 3 * w // 4, y + h // 2)
        }

        for key, (px, py) in points.items():
            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                b, g, r = image[py, px]
                rgb = (r, g, b)
                sampled_colors.append(rgb)
                face_row[key] = str(rgb)
                if show_keypoints:
                    cv2.circle(image, (px, py), 5, (0, 255, 0), -1)
            else:
                face_row[key] = "Out of bounds"

        if sampled_colors:
            avg_r = sum(c[0] for c in sampled_colors) / len(sampled_colors)
            avg_g = sum(c[1] for c in sampled_colors) / len(sampled_colors)
            avg_b = sum(c[2] for c in sampled_colors) / len(sampled_colors)
            avg_lightness = (avg_r + avg_g + avg_b) / 3

            if avg_lightness > 180:
                face_row["Skin Tone"] = "Fair"
            elif avg_lightness > 100:
                face_row["Skin Tone"] = "Medium"
            else:
                face_row["Skin Tone"] = "Dark"

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if show_labels:
            label = f"Face {face_id}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        result.append(face_row)

    return result, image, len(faces)

# ----------- STREAMLIT UI -----------
st.set_page_config("Face RGB + Skin Tone Analyzer", layout="wide")
st.title("🎨 Multi-Face RGB + Skin Tone Analyzer")

uploaded_files = st.file_uploader("📂 Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    show_labels = st.toggle("🧩 Show face number labels on images", value=True)
    show_keypoints = st.toggle("🎯 Show facial RGB keypoints", value=True)

    all_data = []
    zip_buffer = BytesIO()
    face_counts = {}

    with ZipFile(zip_buffer, "w") as zipf:
        for file in uploaded_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            faces, annotated_img, face_count = extract_faces_from_image(image, file.name, show_labels, show_keypoints)
            all_data.extend(faces)
            face_counts[file.name] = face_count

            _, img_encoded = cv2.imencode(".jpg", annotated_img)
            zipf.writestr(f"annotated_{file.name}", img_encoded.tobytes())

            with st.expander(f"🖼️ {file.name} ({face_count} face{'s' if face_count != 1 else ''} detected)"):
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    if all_data:
        df = pd.DataFrame(all_data)

        st.subheader("🎛️ Filter Table by Skin Tone")
        skin_options = ["All"] + sorted(df["Skin Tone"].dropna().unique().tolist())
        selected_tone = st.selectbox("Select Skin Tone", skin_options)

        filtered_df = df.copy()
        if selected_tone != "All":
            filtered_df = df[df["Skin Tone"] == selected_tone]

        st.info(f"Showing {len(filtered_df)} face(s) with tone: {selected_tone}")

        st.subheader("📊 Skin Tone Distribution")
        tone_counts = filtered_df["Skin Tone"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(tone_counts, labels=tone_counts.index, autopct='%1.1f%%', startangle=90,
               colors=["#f8c471", "#f0b27a", "#935116"])
        ax.axis('equal')
        st.pyplot(fig)

        st.subheader("🎨 RGB Preview Table")

        def rgb_span(rgb_string):
            try:
                r, g, b = eval(rgb_string)
                color_hex = f'#{r:02x}{g:02x}{b:02x}'
                return f'<div style="display:flex;align-items:center;">' \
                       f'<div style="width:15px;height:15px;background:{color_hex};margin-right:6px;border-radius:2px;"></div>' \
                       f'<span>{rgb_string}</span></div>'
            except:
                return rgb_string

        color_df = filtered_df.copy()
        for col in ["Forehead RGB", "Left Cheek RGB", "Right Cheek RGB"]:
            color_df[col] = color_df[col].apply(rgb_span)

        st.markdown(color_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        csv_data = df.to_csv(index=False)
        json_data = df.to_json(orient="records", indent=2)

        st.download_button("⬇ Download CSV", csv_data, file_name="face_rgb_skin.csv", mime="text/csv")
        st.download_button("⬇ Download JSON", json_data, file_name="face_rgb_skin.json", mime="application/json")
        st.download_button("⬇ Download Annotated Images (ZIP)",
                           zip_buffer.getvalue(), file_name="annotated_images.zip", mime="application/zip")
    else:
        st.warning("⚠️ No faces detected.")