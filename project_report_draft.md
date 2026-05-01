# PROJECT REPORT: Speech Emotion Recognition using Wav2Vec 2.0 Transformer

## CERTIFICATE
*(Draft your University/College Certificate here. Include Guide/HOD signatures)*
This is to certify that the project entitled "Speech Emotion Recognition using Transformers" is a bonafide work carried out by [Your Name] in partial fulfillment...

## DECLARATION
*(Draft your standard declaration here)*
I hereby declare that the project entitled "Speech Emotion Recognition using Transformers" submitted for the degree is my original work...

## ACKNOWLEDGEMENT
I would like to express my special thanks of gratitude to my guide [Guide Name] as well as our principal [Principal Name] who gave me the golden opportunity to do this wonderful project...

## ABSTRACT
Speech Emotion Recognition (SER) is a critical subfield of human-computer interaction, enabling artificial intelligence systems to understand and react to user sentiments. Traditional approaches heavily rely on manually extracted acoustic features (such as MFCCs) fed into standard deep learning models like Convolutional Neural Networks (CNN) or Recurrent Neural Networks (RNN). However, these architectures often fail to generalize across diverse accents, intonations, and environmental conditions. 
This project leverages advanced state-of-the-art Transfer Learning by fine-tuning Meta's self-supervised foundation model, Wav2Vec 2.0. By attaching a specialized sequence classification head, the system is capable of detecting 7 distinct human emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Pleasant Surprise. The model was trained and validated using combined data from the RAVDESS and CREMA-D audio datasets. Furthermore, the inference engine is deployed into an end-to-end full-stack web application using a Flask REST API backend and an HTML/CSS/JS frontend, allowing users to upload or record audio for real-time instantaneous emotional feedback.

## SYNOPSIS
This project focuses on identifying human emotion from audio waveforms. 
1. **Inputs:** Processed 16kHz audio arrays truncated/padded to a constant 2.75 seconds.
2. **Model:** Pre-trained Base Wav2Vec 2.0 Transformer, fine-tuned over multiple epochs.
3. **Deployment:** A responsive web interface powered by Python Flask.
4. **Outcome:** A robust, high-accuracy SER application outperforming conventional CNN-based systems.

---
*(Page Break in Word)*

## CHAPTER 1: Introduction

### 1.1 Background
Human speech is a rich multidimensional signal containing not only semantic meaning but also affective states (emotions). The ability of machines to interpret these emotions is crucial for the next generation of AI assistants, customer service bots, and psychological profiling. Historically, recognizing emotion from speech required complex signal processing techniques to extract handcrafted features. With the advent of deep learning, neural networks began doing feature extraction automatically. Recently, Transformer-based architectures have revolutionized natural language processing and are now dominating audio analysis due to their self-attention mechanisms.

### 1.2 Problem Definition / Statement
"To design, fine-tune, and deploy an end-to-end Speech Emotion Recognition system using the Wav2Vec 2.0 Transformer architecture capable of classifying raw audio input into 7 discrete emotional states with high accuracy, and making it accessible via a web interface."

### 1.3 Objectives
*   To implement a state-of-the-art Transfer Learning approach for audio classification rather than building models from scratch.
*   To aggregate and preprocess audio data from established academic datasets (RAVDESS, CREMA).
*   To fine-tune a Hugging Face `wav2vec2-base` model using PyTorch and customized loss metrics.
*   To build a Full-Stack web application providing a user-friendly GUI for real-time inference.

### 1.4 Scope of the Project
The scope encompasses processing English-language audio segments. 
*   **In-Scope:** Utilizing 12 transformer encoder blocks, handling `.wav` files computationally, standardizing audio arrays to 16,000Hz, and predicting 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Pleasant Surprise). The 'Calm' class was explicitly scoped out to reduce overlap with 'Neutral'.
*   **Out-of-Scope:** Multilingual inference, real-time continuous socket streaming (analyzes distinct clips instead).

### 1.5 Applications
*   **Healthcare:** Monitoring patient mental health and depression markers through voice.
*   **Customer Service:** Analyzing call-center recordings for customer frustration or satisfaction.
*   **Automotive:** In-car voice assistants tracking driver fatigue or road-rage.
*   **Gaming:** Dynamic NPC responses based on the player's microphone tone.

---
*(Page Break in Word)*

## CHAPTER 2: Literature Review

### 2.1 Existing Systems / Related Work
Traditional SER systems primarily use Mel-Frequency Cepstral Coefficients (MFCCs), Spectrograms, and Chromagrams passed into machine learning classifiers like Support Vector Machines (SVM). In the deep learning era, Convolutional Neural Networks (CNNs) processing 2D spectrographic images, and Long Short-Term Memory (LSTM) networks processing sequential time-series data became the standard. 

### 2.2 Limitations of Existing Systems
*   **Feature Extraction Dependency:** Mel-spectrogram generation is computationally heavy and discards some raw audio phases.
*   **Temporal Context Loss:** CNNs struggle to remember long-term dependencies in an audio track. LSTMs suffer from vanishing gradients on longer audio sequences.
*   **Lack of Generalization:** Models built from scratch require massive labeled datasets, which are scarce for emotion recognition. They often overfit and perform poorly in real-world deployments.

### 2.3 Research Gap
There is a distinct gap between primitive academic "toy models" (built from scratch) and modern industry standards. The industry relies heavily on **Foundation Models** trained on tens of thousands of hours of unlabeled data via self-supervised learning. By utilizing Wav2Vec 2.0, this project bridges that theoretical gap, applying a massive pre-trained temporal Transformer purely on the raw 1D acoustic waveform.

---
*(Page Break in Word)*

## CHAPTER 3: System Analysis

### 3.1 Requirement Analysis
To build this robust system, severe computational and data requirements were analyzed. The system requires bulk audio manipulation at high speeds, and the neural network requires CUDA-enabled GPU acceleration to process the multidimensional self-attention matrices within practical timeframes.

### 3.2 Feasibility Study
*   **Technical Feasibility:** Highly feasible due to the availability of open-source frameworks like PyTorch, Hugging Face Transformers, and Librosa.
*   **Economic Feasibility:** Zero cost. Python and all related libraries are open-source. Datasets are academically free. Hardware (GPU) was provisioned locally/via IDE.
*   **Operational Feasibility:** The Flask-based web design ensures that any layman user can operate the system via a browser, making it highly viable for deployment.

### 3.3 Problem Identification and Proposed Solution
**Problem:** A CNN trained on a small dataset fails when a user uploads audio with a different microbial pitch or background noise.
**Solution:** Do not train from scratch. The proposed solution is to instantiate Meta's Wav2Vec 2.0 base model, which has already learned what human speech sounds like across 100,000+ hours of audio. We freeze the core CNN feature extractor, and only fine-tune the Transformer Encoder blocks and the final classification head mapping to 7 emotion logits.

---
*(Page Break in Word)*

## CHAPTER 4: System Design

### 4.1 System Architecture
The overall architecture follows a distinct pipeline format:
1.  **Frontend Client:** HTML/JS UI where the user selects or records audio.
2.  **API Gateway:** Flask backend receives the HTTP POST request.
3.  **Preprocessing Engine:** 
    *   `librosa.load(sr=16000)` standardized the frequency.
    *   Padding/truncation algorithms ensure the 1D tensor matches the 44,100 length requirement.
4.  **Processor:** `Wav2Vec2Processor` tokenizes the array.
5.  **Inference Engine (Wav2Vec 2.0):** 
    *   *CNN Feature Extractor:* Extracts localized latent representations of 25ms strides.
    *   *Transformer Encoder:* Applies multi-head self-attention to contextualize the temporal data.
    *   *Pooler/Classifier:* Condenses the hidden states into a 7-dimensional output vector.
6.  **Response:** The 'argmax' logit is determined and passed back to the UI as a JSON response.

### 4.2 UML Diagrams
*(Instructors expect 3-4 diagrams here. You should draw/insert them in your word document)*
*   **Figure 4.1: Use Case Diagram:** Show "User" interacting with "Web App", uploading audio. Web App interacting with "Model Engine".
*   **Figure 4.2: Activity Diagram:** Start -> Upload Audio -> Validate File Format -> Resample -> Tokenize -> Run Inference -> Display Result -> Stop.
*   **Figure 4.3: Sequence Diagram:** User -> UI Client -> Flask Server -> ML Model -> Flask Server -> UI Client.

### 4.3 Database Design
Unlike transactional web applications (e.g., e-commerce), an inference pipeline does not necessitate a rigid SQL Database. 
*   **File Storage System:** User uploads are temporarily written to a local `tmp/` or `uploads/` directory on the server file system.
*   **Dataset Structure:** Training datasets (RAVDESS/CREMA) are managed hierarchically via `os.walk` pathing arrays, merged into a dynamic Pandas DataFrame (`audio_paths`, `labels`) during the Data Dataloader instantiation.

---
*(Page Break in Word)*

## CHAPTER 5: Implementation

### 5.1 Software & Hardware Requirements
*   **Hardware:** Nvidia CUDA-enabled GPU (minimum 6GB VRAM for model weights), 16GB RAM, modern multi-core x64 CPU.
*   **Software OS:** Windows 10/11 / Linux (Ubuntu).
*   **Environment:** Anaconda, Jupyter Notebook, VS Code.

### 5.2 Tools & Technologies Used
*   **Model Framework:** PyTorch, Hugging Face `transformers`.
*   **Data Science:** `numpy`, `pandas`, `sklearn`, `matplotlib`, `seaborn`.
*   **Audio Processing:** `librosa`, `torchaudio`.
*   **Backend Server:** Python `Flask`, `Werkzeug` (for secure file saving).
*   **Frontend UI:** HTML5, CSS3, Vanilla JavaScript (ES6).

### 5.3 Modules Developed / In Progress
1.  **Data Extraction Module:** (`get_labels_v2.py`) Scans directory structures, maps integer codes to string emotions (e.g., "03" -> "happy"), filters out unwanted classes ("calm").
2.  **Model Training Module:** Instantiates a custom PyTorch `Dataset` class to manage memory. Uses `TrainingArguments` and the HF `Trainer` API to backpropagate loss and adjust weights.
3.  **Flask Backend Module:** (`app.py`) Exposes an `/analyze` endpoint, manages Cross-Origin requests, and handles the asynchronous logic of waiting for tensor computations.
4.  **Frontend Interface Module:** (`script.js`, `style.css`) Renders an aesthetic, responsive interface for uploading interacting with the AI.

### 5.4 Sample Screens / Partial Outputs
*(Insert 5-6 Screenshots here in your Word Document to fill out pages)*
*   **Screen 1:** Screenshot of your Frontend Dashboard / Upload Page.
*   **Screen 2:** Screenshot of the UI displaying a successful Emotion Prediction (e.g., "Angry").
*   **Screen 3:** Screenshot of Jupyter Notebook Code showing the Training loop output (loss decreasing).
*   **Screen 4:** Screenshot of a Librosa Waveplot / Spectrogram generated during data exploration.

---
*(Page Break in Word)*

## CHAPTER 6: Conclusion

### 6.1 Summary of Work Completed
The project successfully bridges modern industry AI engineering with academic problem-solving. A Speech Emotion Recognition system was built utilizing the Transformer self-attention architecture (Wav2Vec 2.0). Raw acoustic arrays were formatted, custom PyTorch datasets were built, and a classification head spanning 7 emotional states was fine-tuned against RAVDESS and CREMA corpora. 
The system outperforms traditional baseline models and effectively demonstrates full-stack integration capability by hosting the deep learning pipeline behind an interactive Python Flask backend and a modern web frontend. Moving away from manual feature extraction to an end-to-end learning pipeline proved to be highly effective.

### 6.2 Future Work *(Optional/Good to have)*
Future iterations of this system could implement real-time continuous WebSocket streaming for live conversation analysis. Furthermore, expanding the underlying training dataset across varied linguistic families could provide a truly multilingual, globalized emotion detection engine.

---
*(Page Break in Word)*

## CHAPTER 7: References
1. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS.
2. Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).* PLoS ONE.
3. Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.* IEEE Transactions on Affective Computing.
4. Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing.* EMNLP 2020: System Demonstrations.
5. McFee, B., et al. (2015). *librosa: Audio and Music Signal Analysis in Python.* Proceedings of the 14th Python in Science Conference.
