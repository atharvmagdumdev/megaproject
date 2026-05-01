document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('record-btn');
    const recordStatus = document.getElementById('record-status');
    const resultPanel = document.getElementById('result-panel');
    const emotionDisplay = document.getElementById('predicted-emotion');
    const audioUpload = document.getElementById('audio-upload');
    const fileNameDisplay = document.getElementById('file-name');

    let isRecording = false;
    let mediaRecorder;
    let audioChunks = [];

    // Probability Bars
    const emotionBars = {
        'Happy': { label: document.getElementById('prob-happy-val'), bar: document.getElementById('prob-happy') },
        'Sad': { label: document.getElementById('prob-sad-val'), bar: document.getElementById('prob-sad') },
        'Angry': { label: document.getElementById('prob-angry-val'), bar: document.getElementById('prob-angry') },
        'Neutral': { label: document.getElementById('prob-neutral-val'), bar: document.getElementById('prob-neutral') },
        'Fear': { label: document.getElementById('prob-fear-val'), bar: document.getElementById('prob-fear') },
        'Disgust': { label: document.getElementById('prob-disgust-val'), bar: document.getElementById('prob-disgust') },
        'Ps': { label: document.getElementById('prob-ps-val'), bar: document.getElementById('prob-ps') }
    };

    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            // Start Recording
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = e => {
                    audioChunks.push(e.data);
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    sendAudioToBackend(audioBlob, "recording.webm");
                    audioChunks = []; // reset
                };

                mediaRecorder.start();
                isRecording = true;
                recordBtn.classList.add('recording');
                recordStatus.textContent = "Recording... Click to Stop";
                recordStatus.style.color = "var(--accent-glow-secondary)";
                
                // Hide result panel if it was open
                resultPanel.style.display = "none";
                setTimeout(() => resultPanel.classList.add('hidden'), 50);

            } catch (err) {
                console.error("Error accessing mic:", err);
                alert("Please allow microphone access to use this feature.");
            }
        } else {
            // Stop Recording
            mediaRecorder.stop();
            // Stop all tracks to turn off the red dot in the browser tab
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            
            isRecording = false;
            recordBtn.classList.remove('recording');
            recordStatus.textContent = "Analyzing Audio...";
            recordStatus.style.color = "var(--accent-primary)";
        }
    });

    // Handle File Upload Change
    audioUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            fileNameDisplay.textContent = `Selected: ${file.name}`;
            recordStatus.textContent = "Analyzing uploaded file...";
            sendAudioToBackend(file, file.name);
        }
    });

    function sendAudioToBackend(blobOrFile, filename) {
        // Create FormData
        const formData = new FormData();
        formData.append("audio", blobOrFile, filename);

        // Send to Flask
        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                recordStatus.textContent = "Error during analysis.";
                recordStatus.style.color = "var(--color-angry)";
                return;
            }
            showResults(data);
        })
        .catch(err => {
            console.error("Error sending to backend:", err);
            recordStatus.textContent = "Server disconnected.";
            recordStatus.style.color = "var(--color-angry)";
        });
    }

    function showResults(data) {
        // Reset status
        recordStatus.textContent = "Click to Record";
        recordStatus.style.color = "var(--text-muted)";
        
        // Show panel
        resultPanel.style.display = "flex";
        setTimeout(() => resultPanel.classList.remove('hidden'), 50);

        // Update Main Emotion
        emotionDisplay.textContent = data.emotion;
        
        // Pick a color based on emotion
        const emotionColors = {
            'Happy': 'var(--color-happy)',
            'Sad': 'var(--color-sad)',
            'Angry': 'var(--color-angry)',
            'Neutral': 'var(--color-neutral)',
            'Fear': 'var(--color-fear)',
            'Disgust': 'var(--color-disgust)',
            'Ps': 'var(--color-ps)'
        };
        const activeColor = emotionColors[data.emotion] || "#fff";
        emotionDisplay.style.color = activeColor;
        emotionDisplay.style.textShadow = `0 0 30px ${activeColor}`;

        // Update Progress Bars
        for (const [emo, value] of Object.entries(data.probabilities)) {
            if (emotionBars[emo]) {
                const percentage = Math.round(value * 100);
                setTimeout(() => {
                    emotionBars[emo].label.textContent = percentage + "%";
                    emotionBars[emo].bar.style.width = percentage + "%";
                }, 100); // slight delay for smooth CSS transition
            }
        }

        // Update Accuracy Metrics matching the backend confidence
        const accuracyCircle = document.getElementById('accuracy-circle');
        const accuracyVal = document.getElementById('accuracy-val');
        if (accuracyCircle && accuracyVal && data.confidence) {
            const confPercent = Math.round(data.confidence * 100);
            accuracyVal.textContent = confPercent + "%";
            accuracyCircle.style.background = `conic-gradient(${activeColor} ${confPercent}%, rgba(255,255,255,0.1) ${confPercent}%)`;
            accuracyCircle.style.boxShadow = `0 0 30px ${activeColor}`;
        }
    }
});
