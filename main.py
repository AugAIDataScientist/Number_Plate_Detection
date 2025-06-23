import streamlit as st
import cv2
import tempfile
import re
import easyocr
import numpy as np
from ultralytics import YOLO
from pymongo import MongoClient
from datetime import datetime
from groq import Groq
import os
import json

# === Streamlit UI Configuration ===
st.set_page_config(
    page_title="Number Plate Detector", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS for Better UI ===
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .stAlert > div {
        padding: 15px;
        border-radius: 10px;
    }
    .plate-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 5px 0;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        background: #fafafa;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown('<h1 class="main-header">🚘 Smart Number Plate Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image or video to detect number plates with AI-powered recognition</p>', unsafe_allow_html=True)

# === Sidebar Configuration ===
# === Sidebar Configuration ===
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("🔑 Groq API Key", type="password", help="Enter your Groq API key for enhanced user data generation")
    
    st.markdown("---")
    st.header("📊 Session Stats")
    
    # Initialize session state for current session results
    if 'current_session_plates' not in st.session_state:
        st.session_state.current_session_plates = []
    if 'session_stats' not in st.session_state:
        st.session_state.session_stats = {'total_detections': 0, 'unique_plates': 0}

# === MongoDB Setup ===
@st.cache_resource
def init_mongodb():
    try:
<<<<<<< HEAD
        MONGO_URL="mongodb+srv://asadullahmasood1005:o6JMETlQXlGKy8T5@cluster0.nio7sh8.mongodb.net/"
=======
        MONGO_URI ="mongodb+srv://asadullahmasood1005:o6JMETlQXlGKy8T5@cluster0.nio7sh8.mongodb.net/"
>>>>>>> 446586b (usama)
        client = MongoClient(MONGO_URI)
        db = client["car_plate_db"]
        collection = db["plate_records"]
        return collection
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
        return None

collection = init_mongodb()

# === Load Models ===
@st.cache_resource
def load_models():
    try:
        MODEL_PATH = "number_plate_best.pt"
        model = YOLO(MODEL_PATH)
        ocr_reader = easyocr.Reader(['en'])
        return model, ocr_reader
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

model, ocr_reader = load_models()

# === Text Cleanup ===
def extract_valid_text(text_list):
    pattern = r'[A-Za-z0-9]+'
    valid_texts = []
    for text in text_list:
        cleaned = "".join(re.findall(pattern, text)).upper()
        if len(cleaned) >= 3:  # Minimum length for valid plate
            valid_texts.append(cleaned)
    return valid_texts

# === Generate User Info with Groq ===
# === Hardcoded Groq API Key ===
# === Generate User Info with Groq ===
# Remove the hardcoded API key line and use the sidebar input instead

def generate_dummy_info(plate_number):
    try:
        # Use the API key from the sidebar input
        if not groq_api_key:
            st.warning("⚠️ Please enter your Groq API key in the sidebar")
            return None
            
        client = Groq(api_key=groq_api_key)
        prompt = f"""
        Generate realistic Pakistani user data for car plate {plate_number}.
        for every car plate detection it should be unique owner_name
        for every car plate detection it should be unique Phone number
        for every car plate detection it should be unique back card number
        for every car plate detection it can by any one of payment method EasyPaisa or JazzCash or Credit Card,
        Return ONLY valid JSON with these exact keys:
        {{
            "owner_name": "Pakistani name",
            "phone": "03XXXXXXXXX format",
            "bank_card": "**** **** **** XXXX",
            "payment_method": "EasyPaisa or JazzCash or Credit Card"
        }}
        """
        
        # Rest of your function remains the same
        
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You generate Pakistani user data in JSON format only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        reply = chat_completion.choices[0].message.content.strip()
        json_start = reply.find('{')
        json_end = reply.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = reply[json_start:json_end]
            user_data = json.loads(json_str)
            # Validate keys
            required_keys = {"owner_name", "phone", "bank_card", "payment_method"}
            if not required_keys.issubset(user_data.keys()):
                raise ValueError("Groq response missing required fields.")
            return user_data
        else:
            raise ValueError("No valid JSON found in Groq response.")

    except Exception as e:
        st.error(f"❌ Groq API Error: {e}")
        return None

# === Save to MongoDB (No Fallback) ===
def save_plate_to_db(plate_number):
    if collection is None:
        return None
    
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        existing = collection.find_one({"plate_number": plate_number})
        
        if existing:
            collection.update_one(
                {"plate_number": plate_number},
                {"$inc": {"detection_count": 1}, "$set": {"detection_time": now}}
            )
            user_info = {
                "owner_name": existing["owner_name"],
                "phone": existing["phone"],
                "bank_card": existing["bank_card"],
                "payment_method": existing["payment_method"]
            }
        else:
            user_info = generate_dummy_info(plate_number)
            if user_info is None:
                return None  # Groq failed, skip this plate
            
            user_info.update({
                "plate_number": plate_number,
                "detection_time": now,
                "detection_count": 1
            })
            collection.insert_one(user_info)
        
        return user_info
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        return None

# === Image Processing ===
def process_image(image_np):
    if model is None or ocr_reader is None:
        st.error("Models not loaded properly!")
        return image_np, []
    
    detected_plates = []
    results = model(image_np)
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                roi = image_np[y1:y2, x1:x2]
                
                if roi.size > 0:
                    try:
                        ocr_result = ocr_reader.readtext(roi)
                        if ocr_result:
                            texts = [text[1] for text in ocr_result if len(text) > 2 and text[2] > 0.3]  # Confidence threshold
                            plate_texts = extract_valid_text(texts)
                            
                            if plate_texts:
                                plate_text = plate_texts[0]
                                user_info = save_plate_to_db(plate_text)
                                
                                if user_info is not None:
                                    detected_plates.append({
                                        'plate': plate_text,
                                        'user_info': user_info,
                                        'bbox': (x1, y1, x2, y2)
                                    })
                                    
                                    # Draw bounding box and text
                                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    cv2.putText(image_np, plate_text, (x1, y1 - 15), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    except Exception as e:
                        st.warning(f"OCR processing error: {e}")
                        continue
    
    return image_np, detected_plates

# === Video Processing ===
def process_video(input_path, output_path):
    if model is None or ocr_reader is None:
        st.error("Models not loaded properly!")
        return []
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    detected_plates = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        # Process every 10th frame to speed up
        if frame_count % 10 == 0:
            results = model(frame)
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        roi = frame[y1:y2, x1:x2]
                        
                        if roi.size > 0:
                            try:
                                ocr_result = ocr_reader.readtext(roi)
                                if ocr_result:
                                    texts = [text[1] for text in ocr_result if len(text) > 2 and text[2] > 0.3]
                                    plate_texts = extract_valid_text(texts)
                                    
                                    if plate_texts:
                                        plate_text = plate_texts[0]
                                        user_info = save_plate_to_db(plate_text)
                                        
                                        if user_info is not None and plate_text not in [p['plate'] for p in detected_plates]:
                                            detected_plates.append({
                                                'plate': plate_text,
                                                'user_info': user_info
                                            })
                            except Exception as e:
                                continue  # Skip this detection on error
        
        # Draw rectangles for all detected plates in current frame
        results = model(frame)
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "PLATE", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    return detected_plates

# === Main Upload Section ===
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📁 Choose an Image or Video File", 
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    help="Supported formats: JPG, JPEG, PNG for images | MP4, AVI, MOV for videos"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Clear previous session results
    st.session_state.current_session_plates = []
    st.session_state.session_stats = {'total_detections': 0, 'unique_plates': 0}
    
    file_type = uploaded_file.type
    
    if "image" in file_type:
        # === Image Processing ===
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(uploaded_file, caption="Uploaded Image", use_container_width =True)
        
        with col2:
            st.subheader("🔍 Processing Results")
            
            with st.spinner("🔄 Analyzing image..."):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image_np = cv2.imdecode(file_bytes, 1)
                processed_img, detected_plates = process_image(image_np)
            
            if detected_plates:
                st.image(processed_img, caption="Detected Plates", channels="BGR", use_container_width =True)
                st.success(f"✅ Found {len(detected_plates)} number plate(s)!")
                
                # Update session state
                st.session_state.current_session_plates = detected_plates
                st.session_state.session_stats['total_detections'] = len(detected_plates)
                st.session_state.session_stats['unique_plates'] = len(set([p['plate'] for p in detected_plates]))
            else:
                st.warning("⚠️ No number plates detected in this image.")
    
    elif "video" in file_type:
        # === Video Processing ===
        st.subheader("🎥 Video Processing")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(uploaded_file.read())
            input_path = temp_input.name
        
        output_path = "processed_video.mp4"
        
        with st.spinner("🔄 Processing video... This may take a while."):
            detected_plates = process_video(input_path, output_path)
        
        if detected_plates:
            st.success(f"✅ Video processed! Found {len(detected_plates)} unique plates.")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.video(output_path)
            with col2:
                st.metric("Total Detections", len(detected_plates))
                st.metric("Processing Status", "Complete ✅")
            
            # Update session state - limit to first 5 for videos
            st.session_state.current_session_plates = detected_plates[:5]
            st.session_state.session_stats['total_detections'] = len(detected_plates)
            st.session_state.session_stats['unique_plates'] = len(detected_plates)
        else:
            st.warning("⚠️ No number plates detected in this video.")
        
        # Clean up temp file
        try:
            os.unlink(input_path)
        except:
            pass

# === Display Current Session Results ===
if st.session_state.current_session_plates:
    st.markdown("---")
    st.subheader("📋 Current Session Results")
    
    # Display stats in sidebar
    with st.sidebar:
        st.metric("Total Detections", st.session_state.session_stats['total_detections'])
        st.metric("Unique Plates", st.session_state.session_stats['unique_plates'])
    
    # Display detected plates
    for idx, plate_data in enumerate(st.session_state.current_session_plates, 1):
        plate_info = plate_data['user_info']
        
        st.markdown(f"""
        <div class="plate-card">
            <h3>🚗 Plate #{idx}: {plate_data['plate']}</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                <div><strong>👤 Owner:</strong> {plate_info.get('owner_name', 'Unknown')}</div>
                <div><strong>📞 Phone:</strong> {plate_info.get('phone', 'N/A')}</div>
                <div><strong>💳 Bank Card:</strong> {plate_info.get('bank_card', 'N/A')}</div>
                <div><strong>💸 Payment:</strong> {plate_info.get('payment_method', 'N/A')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🔒 Data is securely stored in MongoDB | 🤖 Powered by YOLO & EasyOCR | 🧠 Enhanced by Groq LLaMA</p>
</div>
""", unsafe_allow_html=True)
