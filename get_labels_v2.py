import os
import pandas as pd

crema_path = 'C:/Users/HP/anaconda_projects/megaproject/Datasets/Crema/AudioWAV'
ravdess_path = "C:/Users/HP/anaconda_projects/megaproject/Datasets/RAVDESS"

crema_paths = []
crema_labels = []
emotion_map_crema = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}

if os.path.exists(crema_path):
    for dirname, _, filenames in os.walk(crema_path):
        for filename in filenames:
            parts = filename.split("_")
            if len(parts) > 2:
                emotion_code = parts[2]
                if emotion_code in emotion_map_crema:
                    emotion = emotion_map_crema[emotion_code]
                    if emotion != 'calm':
                        crema_paths.append(os.path.join(dirname, filename))
                        crema_labels.append(emotion)

ravdess_paths = []
ravdess_labels = []
emotion_map_ravdess = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fear", "07": "disgust", "08": "ps"}

if os.path.exists(ravdess_path):
    for dirname, _, filenames in os.walk(ravdess_path):
        for filename in filenames:
            parts = filename.split("-")
            if len(parts) > 2:
                emotion_code = parts[2]
                if emotion_code in emotion_map_ravdess:
                    emotion = emotion_map_ravdess[emotion_code]
                    if emotion != 'calm':
                        ravdess_paths.append(os.path.join(dirname, filename))
                        ravdess_labels.append(emotion)

paths = ravdess_paths + crema_paths
labels = ravdess_labels + crema_labels

df = pd.DataFrame()
df['audio_paths'] = paths
df['labels'] = labels

label_map = {label: idx for idx, label in enumerate(df['labels'].unique())}
inverse_label_map = {idx: label for label, idx in label_map.items()}

print("LIST_FORMAT:", [inverse_label_map[i] for i in range(len(inverse_label_map))])
