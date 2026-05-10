from flask import Flask, render_template, request, jsonify, Response
import os

# Dynamically add the FFmpeg path the user downloaded so it works without them needing to restart terminals or edit Windows PATH variables
ffmpeg_path = r"C:\Users\HP\Downloads\ffmpeg-8.1.1-essentials_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

app = Flask(__name__)


try:
    import cv2
    from deepface import DeepFace
except ImportError:
    print("Please install opencv-python and deepface: pip install opencv-python deepface")
    cv2 = None
    DeepFace = None


print("Loading model and processor, this may take a moment...")
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')


MODEL_DIR = './saved_model' # Placeholder for user to update later
try:
    if os.path.exists(MODEL_DIR):
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=7)
        print(f"Fine-tuned model loaded from {MODEL_DIR}")
    else:
        print("Fine-tuned model not found. Loading base Wav2Vec2 for testing UI...")
        model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base', num_labels=7)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/app")
def model_app():
    return render_template("app.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/face")
def face_app():
    return render_template("face.html")

def generate_frames():
    camera = None
    
    
    sources_to_try = [0, 1, 2, 3]
    
    for source in sources_to_try:
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            success, _ = cap.read()
            if success:
                camera = cap
                print(f"Successfully connected to camera at source {source}")
                break
                
    if camera is None:
        print("Could not find any active camera.")
        return
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                # DeepFace returns a list if multiple faces, or a dict if one
                faces = results if isinstance(results, list) else [results]
                
                for face in faces:
                    # Get bounding box coordinates
                    region = face["region"]
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    emotion = face["dominant_emotion"].capitalize()
                    
                    # Draw a green rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Write the emotion text above the rectangle
                    cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                pass # If detection fails on a frame, we just stream the raw frame
                
            # Convert the modified frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in byte format for the web stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Returns the video stream from the generate_frames() generator
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files["audio"]
    
    # Save the audio to a temporary file
    temp_dir = "tmp"
    os.makedirs(temp_dir, exist_ok=True)
    audio_path = os.path.join(temp_dir, "temp_audio.wav")
    audio_file.save(audio_path)
    
    try:
        if model is None:
            raise ValueError("Model failed to load on server startup.")
            
        # 1. Load the audio file via librosa at 16000 Hz (same as notebook)
        speech, sr = librosa.load(audio_path, sr=16000)

        # 2. Pad or truncate the speech to max_length 44100
        max_length = 44100
        if len(speech) > max_length:
            speech = speech[:max_length]
        else:
            speech = np.pad(speech, (0, max_length - len(speech)), 'constant')

        # 3. Preprocess the audio file
        inputs = processor(
            speech, 
            sampling_rate=16000, 
            return_tensors='pt', 
            padding=True, 
            truncate=True, 
            max_length=max_length
        )
        input_values = inputs.input_values

        # 4. Predict
        with torch.no_grad():
            outputs = model(input_values)
            
        logits = outputs.logits
        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()
        
        predicted_class_idx = logits.argmax(dim=-1).item()
        
        # Safely get label (or fallback if index is out of range)
        predicted_label = emotion_labels[predicted_class_idx] if predicted_class_idx < len(emotion_labels) else f"Class {predicted_class_idx}"
        
        # Format the probabilities dictionary for the frontend
        probabilities_dict = {}
        for idx, p in enumerate(probs):
            lbl = emotion_labels[idx] if idx < len(emotion_labels) else f"Class {idx}"
            probabilities_dict[lbl.capitalize()] = float(p)
            
        result = {
            "emotion": predicted_label.capitalize(),
            "confidence": float(probs[predicted_class_idx]),
            "probabilities": probabilities_dict
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error during prediction: {error_trace}")
        # Fallback to mock data if inference fails (e.g. ffmpeg not installed, etc.)
        result = {
            "emotion": "Error",
            "confidence": 0.0,
            "probabilities": {"Error": 1.0},
            "error_msg": error_trace
        }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
