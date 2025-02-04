import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import cv2  # til video behandling
import folium
from streamlit_folium import st_folium
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def draw_predictions(frame, predictions, confidence_threshold=0.5):
    """Tegner bounding boxes og labels på billedet"""
    frame_with_boxes = frame.copy()
    
    for pred in predictions:
        if pred['probability'] > confidence_threshold:
            # Få bounding box koordinater
            box = pred.get('boundingBox')
            if box:  # Hvis der er bounding box koordinater
                x = int(box['left'] * frame.shape[1])
                y = int(box['top'] * frame.shape[0])
                w = int(box['width'] * frame.shape[1])
                h = int(box['height'] * frame.shape[0])
                
                # Tegn bounding box
                cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Tilføj label med confidence
                label = f"{pred['tagName']}: {pred['probability']:.1%}"
                cv2.putText(frame_with_boxes, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:  # Hvis der ikke er bounding box, vis bare label i toppen
                label = f"{pred['tagName']}: {pred['probability']:.1%}"
                y_position = 30  # Start position for tekst
                cv2.putText(frame_with_boxes, label, (10, y_position),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame_with_boxes

def predict_image(image_data, prediction_key, prediction_endpoint):
    headers = {
        "Content-Type": "application/octet-stream",
        "Prediction-Key": prediction_key
    }
    try:
        response = requests.post(prediction_endpoint, headers=headers, data=image_data)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def process_image(image, prediction_key, prediction_endpoint, enhance=False):
    # Konverter PIL Image til OpenCV format for billedforbedring
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    if enhance:
        # Forbedre billedkvalitet
        enhanced_image = enhance_frame(opencv_image)
        # Konverter tilbage til PIL Image for visning
        display_image = Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    else:
        enhanced_image = opencv_image
        display_image = image
    
    # Konverter det forbedrede billede til bytes for API
    success, encoded_image = cv2.imencode('.jpg', enhanced_image)
    if not success:
        return None
    
    img_byte_arr = encoded_image.tobytes()
    result = predict_image(img_byte_arr, prediction_key, prediction_endpoint)
    
    # Returner både API resultatet og det forbedrede billede
    return result, display_image

def process_video_frame(frame, prediction_key, prediction_endpoint):
    # Konverter frame (som NumPy-array) til JPEG-bytes
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        return None
    image_bytes = encoded_image.tobytes()
    return predict_image(image_bytes, prediction_key, prediction_endpoint)

def show_risk_warning(category, count):
    """Viser en advarsel når der er fundet risikodyr"""
    if category in ["odder", "Stripe Mice"]:
        st.error(f"⚠️ ADVARSEL: Der er fundet {category} ({count} detektioner) ⚠️")
        st.markdown("""
        <div style='background-color: #ffebee; padding: 10px; border-radius: 5px; border-left: 5px solid #f44336;'>
            <h3 style='color: #c62828; margin: 0;'>Risiko Detekteret!</h3>
            <p style='margin: 10px 0;'>Dette område kræver øjeblikkelig opmærksomhed.</p>
        </div>
        """, unsafe_allow_html=True)

def analyze_results(results_data, files_to_process):
    # Opret lister til analyse
    all_predictions = []
    
    # Gennemgå alle resultater
    for result in results_data:
        filename = result['filename']
        predictions = result['predictions']
        
        # Find den prediction med højest sandsynlighed
        if predictions:
            best_prediction = max(predictions, key=lambda x: x['probability'])
            probability = best_prediction['probability'] * 100
            # Hvis prediction er under 50%, kategoriser som "Ikke kategoriseret"
            category = best_prediction['tagName'] if probability >= 50 else "Ikke kategoriseret"
            
            all_predictions.append({
                'Filnavn': filename,
                'Kategori': category,
                'Sikkerhed': probability
            })
    
    if all_predictions:
        # Opret DataFrame
        df = pd.DataFrame(all_predictions)
        
        # Vis samlet statistik
        st.subheader("Samlet Analyse")
        
        # Antal billeder per kategori
        category_counts = df['Kategori'].value_counts()
        
        # Tjek for risikodyr og vis advarsler
        for category, count in category_counts.items():
            show_risk_warning(category, count)
        
        # Opret pie chart med custom farver
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
        # Hvis "Ikke kategoriseret" findes, giv den en grå farve
        if "Ikke kategoriseret" in category_counts.index:
            colors = ['#E6E6E6' if cat == "Ikke kategoriseret" else color 
                     for cat, color in zip(category_counts.index, colors)]
        
        wedges, texts, autotexts = ax1.pie(category_counts.values, 
                                          labels=category_counts.index, 
                                          autopct='%1.1f%%',
                                          colors=colors)
        ax1.axis('equal')
        
        # Tilføj forklaring hvis der er ikke-kategoriserede billeder
        if "Ikke kategoriseret" in category_counts:
            plt.legend(["Billeder med prediction under 50% confidence"])
        
        st.pyplot(fig1)
        
        # Gennemsnitlig sikkerhed per kategori
        avg_confidence = df.groupby('Kategori')['Sikkerhed'].mean().round(2)
        
        # Vis statistik i en tabel
        st.subheader("Statistik per kategori")
        stats_df = pd.DataFrame({
            'Antal Billeder': category_counts,
            'Gennemsnitlig Sikkerhed (%)': avg_confidence
        }).round(2)
        st.dataframe(stats_df)
        
        # Vis alle billeder og deres predictions i én dropdown
        with st.expander("Vis alle analyserede billeder", expanded=False):
            st.write("Billede-for-billede analyse:")
            
            # Vis 3 billeder per række
            for i in range(0, len(results_data), 3):
                # Opret en række med 3 kolonner
                cols = st.columns(3)
                
                # Fyld rækken med op til 3 billeder
                for j in range(3):
                    if i + j < len(results_data):
                        with cols[j]:
                            result = results_data[i + j]
                            filename = result['filename']
                            predictions = result['predictions']
                            
                            st.write(f"### {filename}")
                            
                            # Find og vis det originale billede
                            image = Image.open(next(f for f in files_to_process if f.name == filename))
                            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            
                            # Tegn bounding boxes på billedet
                            image_with_boxes = draw_predictions(opencv_image, predictions)
                            image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                            st.image(image_rgb, caption=f"Detektioner i {filename}", use_container_width=True)
                            
                            # Vis predictions tabel
                            if predictions:
                                pred_df = pd.DataFrame(predictions)
                                # Konverter probability til procent
                                pred_df['probability'] = pred_df['probability'].apply(lambda x: f"{x*100:.1f}%")
                                # Omdøb kolonner til dansk
                                pred_df = pred_df.rename(columns={
                                    'tagName': 'Kategori',
                                    'probability': 'Sikkerhed'
                                })
                                st.dataframe(pred_df[['Kategori', 'Sikkerhed']], use_container_width=True)
                
                # Tilføj separator efter hver række
                st.markdown("---")
    else:
        st.warning("Ingen objekter blev detekteret i billederne med den valgte confidence threshold")

def analyze_video_results(video_predictions, processed_frames, video_name):
    if not video_predictions:
        return
    
    # Samlet analyse af alle predictions
    all_detections = []
    for pred in video_predictions:
        for p in pred['predictions']:
            all_detections.append({
                'frame': pred['filename'],
                'tag': p['tagName'],
                'confidence': p['probability'] * 100
            })
    
    # Konverter til DataFrame for lettere analyse
    df = pd.DataFrame(all_detections)
    
    # Find det mest detekterede objekt
    if not df.empty:
        detection_counts = df['tag'].value_counts()
        most_common_object = detection_counts.index[0]
        detection_percentage = (detection_counts[most_common_object] / len(video_predictions)) * 100
        
        # Tjek for risikodyr og vis advarsler
        for category, count in detection_counts.items():
            show_risk_warning(category, count)
        
        # Vis samlet konklusion
        st.header(f"Analyse af: {video_name}")
        st.write(f"Baseret på analysen af videoen:")
        st.write(f"- Det mest detekterede objekt var: **{most_common_object}**")
        st.write(f"- Dette objekt blev fundet i {detection_percentage:.1f}% af de analyserede frames")
        
        # Vis detaljeret statistik
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Antal detektioner per kategori:")
            st.dataframe(detection_counts)
        
        with col2:
            # Gennemsnitlig confidence per kategori
            avg_confidence = df.groupby('tag')['confidence'].mean().round(2)
            st.write("Gennemsnitlig sikkerhed per kategori:")
            st.dataframe(avg_confidence)
        
        # Vis alle frames i én dropdown
        with st.expander("Vis alle analyserede frames", expanded=False):
            st.write("Frames med detektioner:")
            
            # Vis frames i et grid
            for i in range(0, len(processed_frames), 2):  # Ændret fra 3 til 2 kolonner
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(processed_frames):
                        with cols[j]:
                            frame_idx = i + j
                            frame = processed_frames[frame_idx]
                            predictions = video_predictions[frame_idx]['predictions']
                            
                            # Konverter frame til RGB og tilføj labels
                            frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
                            
                            # Tilføj predictions som tekst på billedet
                            for idx, pred in enumerate(predictions):
                                if pred['probability'] > 0.5:  # Vis kun predictions over 50%
                                    text = f"{pred['tagName']}: {pred['probability']:.1%}"
                                    y_position = 30 + (idx * 30)  # Forskyd tekst vertikalt
                                    cv2.putText(frame_rgb, text, (10, y_position), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Vis frame med labels
                            st.image(frame_rgb, caption=f"Frame {frame_idx + 1}", use_container_width=True)
                            
                            # Vis predictions tabel under billedet
                            if predictions:
                                pred_df = pd.DataFrame(predictions)
                                # Konverter probability til procent
                                pred_df['probability'] = pred_df['probability'].apply(lambda x: f"{x*100:.1f}%")
                                # Omdøb kolonner til dansk
                                pred_df = pred_df.rename(columns={
                                    'tagName': 'Kategori',
                                    'probability': 'Sikkerhed'
                                })
                                st.dataframe(pred_df[['Kategori', 'Sikkerhed']], 
                                           use_container_width=True)
        
        # Tilføj en separator mellem videoer
        st.markdown("---")

def enhance_frame(frame):
    """Forbedrer kvaliteten af en video frame ved hjælp af forskellige teknikker"""
    try:
        # Konverter til float32 for bedre præcision i beregningerne
        frame_float = frame.astype(np.float32) / 255.0

        # 1. Forøg kontrast og lysstyrke
        alpha = 1.3  # Kontrast faktor
        beta = 0.1   # Lysstyrke faktor
        enhanced = cv2.convertScaleAbs(frame_float, alpha=alpha, beta=beta)

        # 2. Reducer støj med bilateral filter (bevarer kanter)
        denoised = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        # 3. Forøg skarphed
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # 4. Juster farver
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_bgr

    except Exception as e:
        st.error(f"Fejl ved billedforbedring: {str(e)}")
        return frame  # Returner original frame hvis der opstår fejl

def process_single_video(video_file, frames_to_skip, confidence_threshold, prediction_key, prediction_endpoint, enhance_frames=False):
    # Gem video midlertidigt
    tfile = io.BytesIO(video_file.read())
    temp_video_path = f"temp_video_{video_file.name}.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(tfile.getvalue())
    
    # Åbn video
    cap = cv2.VideoCapture(temp_video_path)
    frame_count = 0
    processed_frames = []
    video_predictions = []
    
    # Progress bar og status
    progress_text = st.empty()
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Opdater progress
        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Behandler {video_file.name} - Frame {frame_count+1} af {total_frames}")
        
        if frame_count % frames_to_skip == 0:
            # Anvend billedforbedring hvis aktiveret
            if enhance_frames:
                frame = enhance_frame(frame)
            
            result = process_video_frame(frame, prediction_key, prediction_endpoint)
            if result:
                predictions = result.get("predictions", [])
                high_confidence_preds = [p for p in predictions if p["probability"] > confidence_threshold]
                
                if high_confidence_preds:
                    processed_frames.append(frame)
                    video_predictions.append({
                        'filename': f"{video_file.name}_frame_{frame_count}",
                        'predictions': predictions
                    })
        
        frame_count += 1
    
    cap.release()
    progress_text.empty()
    progress_bar.empty()
    
    # Ryd op
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    
    return video_predictions, processed_frames

def save_location_results(lat, lon, results, is_video=False):
    if 'map_data' not in st.session_state:
        st.session_state.map_data = []
    
    # Find det mest almindelige objekt og dets antal
    all_predictions = []
    for result in results:
        predictions = result['predictions']
        if predictions:
            for pred in predictions:
                if pred['probability'] > 0.5:  # Kun medtag predictions over 50%
                    all_predictions.append(pred['tagName'])
    
    if all_predictions:
        most_common = max(set(all_predictions), key=all_predictions.count)
        count = all_predictions.count(most_common)
        
        # Gem resultatet
        st.session_state.map_data.append({
            'lat': lat,
            'lon': lon,
            'object': most_common,
            'count': count,
            'type': 'video' if is_video else 'image',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

def main():
    st.title("Musedetekterings App")
    
    # Azure Custom Vision konstanter fra miljøvariable
    PREDICTION_KEY = os.getenv("PREDICTION_KEY")
    PROJECT_ID = os.getenv("PROJECT_ID")
    PREDICTION_ENDPOINT = f"https://ivmai-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/Iteration5/image"
    
    if not PREDICTION_KEY or not PROJECT_ID:
        st.error("API nøgler ikke fundet. Sørg for at .env filen er korrekt konfigureret.")
        return
    
    # Konfiguration i sidebar
    st.sidebar.header("Konfiguration")
    enhance_media = st.sidebar.checkbox("Aktiver billedforbedring", value=True, 
                                     help="Forbedrer kvaliteten af billeder og video frames")
    
    # Video-specifikke indstillinger
    st.sidebar.subheader("Video Indstillinger")
    frames_to_skip = st.sidebar.slider("Analyser hver N'te frame", min_value=1, max_value=30, value=10)
    confidence_threshold = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.5)
    
    # Initialiser session state for lokationer hvis den ikke findes
    if 'locations' not in st.session_state:
        st.session_state.locations = []
    
    # Initialiser kort centreret på Danmark
    st.header("Vælg Lokationer")
    st.write("1. Zoom og naviger til den ønskede lokation på kortet")
    st.write("2. Klik på kortet for at vælge et punkt")
    st.write("3. Upload filer til den valgte lokation")
    st.write("4. Gentag for flere lokationer eller start analysen")
    
    # Opret kort med folium
    m = folium.Map(location=[56.0, 10.0], zoom_start=7)  # Centreret på Danmark
    
    # Tilføj eksisterende markører til kortet
    if 'map_data' in st.session_state:
        for data in st.session_state.map_data:
            # Tjek om det er et risikodyr
            is_risk = data['object'] in ["odder", "Stripe Mice"]
            
            # Vælg markør farve og ikon baseret på type og risiko
            if is_risk:
                # Tilføj pulserende cirkel for risikodyr
                folium.CircleMarker(
                    location=[data['lat'], data['lon']],
                    radius=15,  # Reduceret fra 30 til 15
                    color='red',
                    fill=True,
                    popup='⚠️ RISIKO OMRÅDE ⚠️',
                    weight=2,
                    fill_opacity=0.3
                ).add_to(m)
                
                # Brug et stort advarselsikon for risikodyr
                icon = folium.Icon(
                    color='red',
                    icon='warning-sign',
                    prefix='glyphicon',
                    icon_size=(40, 40)
                )
                popup_content = f"""
                <div style='color: red; font-weight: bold; min-width: 200px; text-align: center;'>
                    <h3 style='color: red;'>⚠️ RISIKO DETEKTERET! ⚠️</h3>
                    <hr>
                    <b>Type:</b> {data['type']}<br>
                    <b>Objekt:</b> {data['object']}<br>
                    <b>Antal:</b> {data['count']}<br>
                    <b>Tid:</b> {data['timestamp']}<br>
                    <hr>
                    <div style='background-color: #ffebee; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                        <span style='color: darkred; font-size: 14px;'>
                        Dette område kræver øjeblikkelig opmærksomhed!
                        </span>
                    </div>
                </div>
                """
            else:
                # Normal markør for andre dyr
                icon = folium.Icon(
                    color='blue' if data['type'] == 'image' else 'purple',
                    icon='camera' if data['type'] == 'image' else 'film',
                    prefix='glyphicon'
                )
                popup_content = f"""
                <div style='min-width: 200px;'>
                    <b>Type:</b> {data['type']}<br>
                    <b>Objekt:</b> {data['object']}<br>
                    <b>Antal:</b> {data['count']}<br>
                    <b>Tid:</b> {data['timestamp']}
                </div>
                """
            
            # Tilføj markør med popup
            folium.Marker(
                [data['lat'], data['lon']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=icon
            ).add_to(m)
    
    # Tilføj markører for valgte lokationer
    for idx, loc in enumerate(st.session_state.locations):
        folium.Marker(
            [loc['lat'], loc['lng']],
            popup=f"Lokation {idx + 1}",
            icon=folium.Icon(color='green')
        ).add_to(m)
    
    # Vis kortet og få klik-position
    map_data = st_folium(m, height=400, width=700)
    
    # Håndter ny lokation
    if map_data['last_clicked']:
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lng = map_data['last_clicked']['lng']
        st.write(f"Valgt position: {clicked_lat:.4f}, {clicked_lng:.4f}")
        
        # Tilføj lokation knap
        if st.button("Tilføj Lokation"):
            new_location = {
                'lat': clicked_lat,
                'lng': clicked_lng,
                'images': [],
                'videos': []
            }
            st.session_state.locations.append(new_location)
            st.rerun()
    
    # Vis eksisterende lokationer og deres uploads
    if st.session_state.locations:
        st.header("Valgte Lokationer")
        
        for idx, location in enumerate(st.session_state.locations):
            with st.expander(f"Lokation {idx + 1}: {location['lat']:.4f}, {location['lng']:.4f}", expanded=True):
                # Upload billeder
                uploaded_images = st.file_uploader(
                    "Vælg billeder",
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=True,
                    key=f"image_upload_{idx}"
                )
                
                # Upload videoer
                uploaded_videos = st.file_uploader(
                    "Vælg videoer",
                    type=['mp4', 'avi', 'mov'],
                    accept_multiple_files=True,
                    key=f"video_upload_{idx}"
                )
                
                # Gem uploads i session state
                location['images'] = uploaded_images if uploaded_images else []
                location['videos'] = uploaded_videos if uploaded_videos else []
                
                # Vis antal filer
                if uploaded_images or uploaded_videos:
                    st.success(f"Upload succesfuld! {len(uploaded_images)} billede(r) og {len(uploaded_videos)} video(er) uploadet.")
                
                # Tilføj slet lokation knap
                if st.button("Slet Lokation", key=f"delete_{idx}"):
                    st.session_state.locations.pop(idx)
                    st.rerun()
        
        # Tilføj analyse knap når der er mindst én lokation med filer
        has_files = any(len(loc['images']) > 0 or len(loc['videos']) > 0 for loc in st.session_state.locations)
        
        if has_files:
            if st.button("Start Analyse af Alle Lokationer", type="primary"):
                # Gem analyseresultater i session state hvis de ikke findes
                if 'analysis_complete' not in st.session_state:
                    st.session_state.analysis_complete = False
                
                if not st.session_state.analysis_complete:
                    # Analyser hver lokation
                    for location in st.session_state.locations:
                        st.header(f"Analyserer lokation: {location['lat']:.4f}, {location['lng']:.4f}")
                        
                        # Analyser billeder
                        if location['images']:
                            st.subheader("Billedanalyse")
                            results_data = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, file in enumerate(location['images']):
                                try:
                                    status_text.text(f'Analyserer billede {i+1} af {len(location["images"])}: {file.name}')
                                    progress_bar.progress((i + 1) / len(location['images']))
                                    
                                    image = Image.open(file)
                                    result, _ = process_image(image, PREDICTION_KEY, PREDICTION_ENDPOINT, enhance=enhance_media)
                                    
                                    if result:
                                        results_data.append({
                                            'filename': file.name,
                                            'predictions': result['predictions']
                                        })
                                except Exception as e:
                                    st.error(f"Fejl ved behandling af {file.name}: {str(e)}")
                            
                            status_text.empty()
                            progress_bar.empty()
                            
                            if results_data:
                                analyze_results(results_data, location['images'])
                                save_location_results(location['lat'], location['lng'], results_data, is_video=False)
                        
                        # Analyser videoer
                        if location['videos']:
                            st.subheader("Videoanalyse")
                            for video_file in location['videos']:
                                st.write(f"Analyserer: {video_file.name}")
                                
                                video_predictions, processed_frames = process_single_video(
                                    video_file,
                                    frames_to_skip,
                                    confidence_threshold,
                                    PREDICTION_KEY,
                                    PREDICTION_ENDPOINT,
                                    enhance_frames=enhance_media
                                )
                                
                                if video_predictions:
                                    analyze_video_results(video_predictions, processed_frames, video_file.name)
                                    save_location_results(location['lat'], location['lng'], video_predictions, is_video=True)
                        
                        st.markdown("---")
                    
                    # Marker at analysen er færdig
                    st.session_state.analysis_complete = True
                    st.success("Analyse af alle lokationer er gennemført!")
                    
                    # Tilføj en knap til at starte ny analyse
                    if st.button("Start Ny Analyse"):
                        st.session_state.locations = []
                        st.session_state.analysis_complete = False
                        st.rerun()
        else:
            st.warning("Upload filer til mindst én lokation for at starte analysen")

if __name__ == "__main__":
    main()
