import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio

import torchaudio
import torch 
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Trainer, TrainingArguments, Wav2Vec2ForSequenceClassification

import warnings
warnings.filterwarnings('ignore')

import torch
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")


import torch
import torchaudio

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Torchaudio:", torchaudio.__version__)

crema_path= ('C:/Users/HP/anaconda_projects/megaproject/Datasets/Crema/AudioWAV')


crema_paths = []
crema_labels = []

emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

for dirname, _, filenames in os.walk(crema_path):
    for filename in filenames:
        
        parts = filename.split("_")
        emotion_code = parts[2]
        emotion = emotion_map[emotion_code]
        if emotion != 'calm':
            crema_paths.append(os.path.join(dirname, filename))
            crema_labels.append(emotion)
        
print('Dataset is loaded')

'''tess_path = "C:/Users/HP/anaconda_projects/megaproject/Datasets/TESS/TESS Toronto emotional speech set data"

tess_paths = []
tess_labels = []
for dirname, _, filenames in os.walk(tess_path):
    for filename in filenames:
        tess_paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label= label.split('.')[0]
        tess_labels.append(label.lower())
        
print('Dabtaset is loaded')'''

ravdess_path = "C:/Users/HP/anaconda_projects/megaproject/Datasets/RAVDESS"

ravdess_paths = []
ravdess_labels = []

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "ps"
}

for dirname, _, filenames in os.walk(ravdess_path):
    for filename in filenames:
        parts = filename.split("-")
        emotion_code = parts[2]
        emotion = emotion_map[emotion_code]
        if emotion != 'calm':
            ravdess_paths.append(os.path.join(dirname, filename))
            ravdess_labels.append(emotion)

print('Dataset is loaded')

paths =  ravdess_paths + crema_paths #+ tess_paths
labels =  ravdess_labels + crema_labels #+ tess_labels

paths[:3]

labels[:5]

## creating a dataframe
df = pd.DataFrame()
df['audio_paths'] = paths
df['labels'] = labels
df.head()

sns.countplot(data=df, x='labels')

df['labels'].value_counts()

def waveplot(data, sr, emotion):
    plt.figure(figsize = (10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr = sr)
    plt.show()

def spectogram(data, sr, emotion):
    x= librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize = (10, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb , sr = sr, x_axis = 'time', y_axis= 'hz')
    plt.colorbar()

emotion = 'fear'
path = np.array(df['audio_paths'][df['labels'] == emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'angry'
path = np.array(df['audio_paths'][df['labels'] == emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'disgust'
path = np.array(df['audio_paths'][df['labels'] == emotion])[2]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'neutral'
path = np.array(df['audio_paths'][df['labels'] == emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'sad'
path = np.array(df['audio_paths'][df['labels'] == emotion])[2]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'ps'
path = np.array(df['audio_paths'][df['labels'] == emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'happy'
path = np.array(df['audio_paths'][df['labels'] == emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

#convert labels to integers
label_map = {label: idx for idx, label in enumerate(df['labels'].unique())}
inverse_label_map = {idx: label for label, idx in label_map.items()}
df['labels'] = df['labels'].map(label_map)
df.head(2)

class SpeechEmotionDataset(Dataset):
    def __init__(self, df, processor, max_length = 44100):
        self.df= df
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['audio_paths']
        label = self.df.iloc[idx]['labels']

        #load the audio file
        speech,sr = librosa.load(audio_path, sr=16000)

        #pad or truncade the speech to required length
        if len(speech)>self.max_length:
                speech = speech[:self.max_length]
        else:
            speech = np.pad(speech, (0, self.max_length - len(speech)), 'constant')

        #preprocessing the audio file
        inputs = self.processor(speech, sampling_rate = 16000, return_tensors = 'pt', padding = True, truncate = True, max_length = self.max_length)

        input_values = inputs.input_values.squeeze()
        return{'input_values': input_values, 'labels': torch.tensor(label, dtype = torch.long)}

#split The data for train and test
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['labels']
)

#initiating the processor and model
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base', num_labels = 7)

#load the dataset
train_dataset = SpeechEmotionDataset(train_df,processor)
test_dataset = SpeechEmotionDataset(test_df,processor)

train_dataset[0]['input_values'].size()

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy = 'epoch',
    save_strategy= 'epoch',
    learning_rate = 2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay = 0.01,
    report_to=[]
)

#create functions for computing metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids #original labels
    preds = np.argmax(pred.predictions,axis=1) #model predicted labels
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average= 'weighted')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

#initialize the trainer
trainer = Trainer(
    model = model, 
    args = training_args,
    train_dataset =train_dataset,
    eval_dataset =test_dataset,
    compute_metrics = compute_metrics
)
trainer.train()

results = trainer.evaluate()
print(results)

import random
idx = random.randrange(0, len(test_dataset))
print("Original Label:", inverse_label_map[int(test_dataset[idx]['labels'])])
input_values = test_dataset[idx]['input_values'].unsqueeze(0).to('cuda')

with torch.no_grad():
    outputs = model(input_values)
logits = outputs.logits

predicted_class = logits.argmax(dim=-1).item()
print('Predicted Label:',inverse_label_map[predicted_class])

idx = random.randrange(0, len(test_dataset))
print("Original Label:", inverse_label_map[int(test_dataset[idx]['labels'])])
input_values = test_dataset[idx]['input_values'].unsqueeze(0).to('cuda')

with torch.no_grad():
    outputs = model(input_values)
logits = outputs.logits

predicted_class = logits.argmax(dim=-1).item()
print('Predicted Label:',inverse_label_map[predicted_class])

test_path='C:/Users/HP/anaconda_projects/megaproject/Datasets/RAVDESS/Actor_02/03-01-01-01-01-02-02.wav'

class SpeechEmotionDataset(Dataset):
    def __init__(self, df, processor, max_length = 44100):
        self.df= df
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['audio_paths']
        speech,sr = librosa.load(test_path, sr=16000)

        #pad or truncade the speech to required length
        if len(speech)>self.max_length:
                speech = speech[:self.max_length]
        else:
            speech = np.pad(speech, (0, self.max_length - len(speech)), 'constant')

        #preprocessing the audio file
        inputs = self.processor(speech, sampling_rate = 16000, return_tensors = 'pt', padding = True, truncate = True, max_length = self.max_length)

        input_values = inputs.input_values.squeeze()
input_values = test_dataset[idx]['input_values'].unsqueeze(0).to('cuda')

with torch.no_grad():
    outputs = model(input_values)
logits = outputs.logits

predicted_class = logits.argmax(dim=-1).item()
print('Predicted Label:',inverse_label_map[predicted_class])



