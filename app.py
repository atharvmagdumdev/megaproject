from flask import Flask, render_template, request, jsonify
import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

app = Flask(__name__)

# Load the processor (same as in the training notebook)
print("Loading model and processor, this may take a moment...")
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')

# The user is supposed to train and save the model to a path (e.g. ./results/). 
# For now we will instantiate the base model or try to load a local ./model if it exists.
# We set num_labels=7 to match angry, disgust, fear, happy, neutral, sad, ps.
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

# Fallback label map (since the unique() order in pandas might differ, 
# the user will need to supply the exact inverse_label_map generated in their notebook).
# For now, we use a generic mapping for demonstration based on the 7 emotions.
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
        print(f"Error during prediction: {e}")
        # Fallback to mock data if inference fails (e.g. ffmpeg not installed, etc.)
        result = {
            "emotion": "Error",
            "confidence": 0.0,
            "probabilities": {"Error": 1.0},
            "error_msg": str(e)
        }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
