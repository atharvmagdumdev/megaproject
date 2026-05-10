document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('record-btn');
    const recordStatus = document.getElementById('record-status');
    const emotionDisplay = document.getElementById('predicted-emotion');
    const audioUpload = document.getElementById('audio-upload');
    const fileNameDisplay = document.getElementById('file-name');
    const idleHero = document.getElementById('idle-hero');
    const dashboardSection = document.getElementById('dashboard-section');
    
    // Meter elements
    const accuracyVal = document.getElementById('accuracy-val');
    const accuracyBadge = document.getElementById('accuracy-badge');
    const meterPointer = document.getElementById('meter-pointer');
    const meterAiText = document.getElementById('ai-insight-text-meter');
    const emotionIcon = document.getElementById('emotion-icon');
    
    // Insight elements
    const aiInsightText = document.getElementById('ai-insight-text');
    const tagEnergy = document.getElementById('tag-energy');
    const tagTone = document.getElementById('tag-tone');
    const tagPace = document.getElementById('tag-pace');

    let isRecording = false;
    let mediaRecorder;
    let audioChunks = [];
    let globalAudioContext = null;

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
        if (!globalAudioContext) {
            globalAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (globalAudioContext.state === 'suspended') {
            await globalAudioContext.resume();
        }

        if (!isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = e => {
                    audioChunks.push(e.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    audioChunks = [];
                    
                    try {
                        const arrayBuffer = await audioBlob.arrayBuffer();
                        const audioBuffer = await globalAudioContext.decodeAudioData(arrayBuffer);
                        const wavBlob = audioBufferToWav(audioBuffer);
                        sendAudioToBackend(wavBlob, "recording.wav");
                    } catch (err) {
                        console.error("Audio conversion failed:", err);
                        sendAudioToBackend(audioBlob, "recording.webm");
                    }
                };

                mediaRecorder.start();
                isRecording = true;
                recordBtn.classList.add('recording');
                recordStatus.textContent = "Recording... Click to Stop";
                recordStatus.style.color = "var(--accent-glow-secondary)";

                if (idleHero) {
                    idleHero.style.display = "none";
                    idleHero.classList.add('hidden');
                }
                if (dashboardSection) {
                    dashboardSection.style.display = "none";
                    dashboardSection.classList.add('hidden');
                }
            } catch (err) {
                console.error("Error accessing mic:", err);
                alert("Please allow microphone access to use this feature.");
            }
        } else {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            isRecording = false;
            recordBtn.classList.remove('recording');
            recordStatus.textContent = "Analyzing Audio...";
            recordStatus.style.color = "var(--accent-primary)";

            if (dashboardSection) {
                dashboardSection.style.display = "flex";
                setTimeout(() => {
                    dashboardSection.classList.remove('hidden');
                    dashboardSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
            }
            if (emotionDisplay) emotionDisplay.textContent = "Analyzing...";
            if (accuracyVal) accuracyVal.textContent = "0%";
            if (meterPointer) meterPointer.style.left = "50%";
            if (meterAiText) meterAiText.textContent = "Processing audio signals...";
            if (emotionIcon) emotionIcon.textContent = "⏳";
        }
    });

    audioUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            fileNameDisplay.textContent = `Selected: ${file.name}`;
            recordStatus.textContent = "Analyzing uploaded file...";
            
            if (idleHero) {
                idleHero.style.display = "none";
                idleHero.classList.add('hidden');
            }
            if (dashboardSection) {
                dashboardSection.style.display = "flex";
                setTimeout(() => {
                    dashboardSection.classList.remove('hidden');
                    dashboardSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
            }
            if (emotionDisplay) emotionDisplay.textContent = "Analyzing...";
            if (accuracyVal) accuracyVal.textContent = "0%";
            if (meterPointer) meterPointer.style.left = "50%";
            if (meterAiText) meterAiText.textContent = "Processing audio signals...";
            if (emotionIcon) emotionIcon.textContent = "⏳";

            sendAudioToBackend(file, file.name);
        }
    });

    function sendAudioToBackend(blobOrFile, filename) {
        const formData = new FormData();
        formData.append("audio", blobOrFile, filename);
        
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
            if (data.emotion === "Error") {
                console.error("Backend Error:", data.error_msg);
                alert("Backend Error: " + data.error_msg);
                recordStatus.textContent = "Processing Error.";
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
        recordStatus.textContent = "Click to Record";
        recordStatus.style.color = "var(--text-muted)";

        if (dashboardSection) {
            dashboardSection.style.display = "flex";
            setTimeout(() => dashboardSection.classList.remove('hidden'), 50);
        }

        emotionDisplay.textContent = data.emotion;

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

        const insights = {
            'Happy': { icon: '😊', text: 'The vocal tone indicates a higher pitch and faster tempo, characteristic of joy or excitement.', short: 'Detected positive emotional lift and increased vocal energy.', energy: 'High', tone: 'Bright', pace: 'Fast' },
            'Sad': { icon: '😔', text: 'The vocal tone indicates a lower pitch and slower tempo, characteristic of sadness or fatigue.', short: 'Detected lowered pitch variations typical of sadness or fatigue.', energy: 'Low', tone: 'Dark', pace: 'Slow' },
            'Angry': { icon: '😠', text: 'High energy and intensity detected, with sharp vocal inflections characteristic of anger or frustration.', short: 'Detected sharp vocal inflections and harsh tonal intensity.', energy: 'Very High', tone: 'Harsh', pace: 'Fast' },
            'Neutral': { icon: '😐', text: 'The vocal tone is steady and balanced, indicating a calm or neutral emotional state.', short: 'Detected stable vocal frequency with low emotional variance.', energy: 'Medium', tone: 'Even', pace: 'Moderate' },
            'Fear': { icon: '😨', text: 'Tremors or high-pitched variations detected, typical of anxiety or fear.', short: 'Detected vocal tremors or pitch instability linked to anxiety.', energy: 'High', tone: 'Unsteady', pace: 'Fast' },
            'Disgust': { icon: '🤢', text: 'Irregular vocal patterns and lower pitch detected, characteristic of aversion or disgust.', short: 'Detected irregular vocal pacing characteristic of aversion.', energy: 'Low', tone: 'Harsh', pace: 'Slow' },
            'Ps': { icon: '😲', text: 'Sudden spikes in pitch and energy detected, indicating pleasant surprise.', short: 'Detected sudden acoustic spikes indicative of surprise.', energy: 'High', tone: 'Bright', pace: 'Fast' }
        };

        const insight = insights[data.emotion] || { icon: '😐', text: 'Analysis complete. Pattern recognized.', short: 'Analysis complete. Pattern recognized.', energy: '--', tone: '--', pace: '--' };
        
        if (emotionIcon) emotionIcon.textContent = insight.icon;
        if (aiInsightText) aiInsightText.textContent = insight.text;
        if (meterAiText) meterAiText.textContent = insight.short;
        if (tagEnergy) tagEnergy.textContent = "Energy: " + insight.energy;
        if (tagTone) tagTone.textContent = "Tone: " + insight.tone;
        if (tagPace) tagPace.textContent = "Pace: " + insight.pace;

        for (const [emo, value] of Object.entries(data.probabilities)) {
            if (emotionBars[emo]) {
                const percentage = Math.round(value * 100);
                setTimeout(() => {
                    emotionBars[emo].label.textContent = percentage + "%";
                    emotionBars[emo].bar.style.width = percentage + "%";
                }, 100);
            }
        }

        const meterPositions = {
            'Happy': '0%',
            'Ps': '16.6%',
            'Neutral': '33.3%',
            'Sad': '50%',
            'Fear': '66.6%',
            'Disgust': '83.3%',
            'Angry': '100%'
        };
        if (meterPointer) {
            meterPointer.style.left = meterPositions[data.emotion] || '50%';
        }

        if (accuracyVal && data.confidence) {
            const confPercent = Math.round(data.confidence * 100);
            accuracyVal.textContent = confPercent + "%";
            
            if (accuracyBadge) {
                accuracyBadge.style.borderColor = activeColor;
                accuracyBadge.style.boxShadow = `0 0 15px ${activeColor}40`;
            }
        }
    }
});

// Utility function to convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    const result = new Int16Array(buffer.length * numChannels);
    for (let channel = 0; channel < numChannels; channel++) {
        const channelData = buffer.getChannelData(channel);
        let offset = channel;
        for (let i = 0; i < buffer.length; i++) {
            let sample = Math.max(-1, Math.min(1, channelData[i]));
            result[offset] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            offset += numChannels;
        }
    }
    
    const dataSize = result.length * 2;
    const arrayBuffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(arrayBuffer);
    
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');
    
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true);
    view.setUint16(32, numChannels * 2, true);
    view.setUint16(34, bitDepth, true);
    
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);
    
    const offset = 44;
    for (let i = 0; i < result.length; i++) {
        view.setInt16(offset + (i * 2), result[i], true);
    }
    
    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}
