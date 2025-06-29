#REAL TIME KEY LOGGER DETECTION AND ALERT


🔐 Real-Time Keylogger Detection and Alert
A hybrid cybersecurity, AI and blockchain-based system to detect and log keylogger threats in real-time.

📌 Overview
Real-Time Keylogger Detection and Alert is a robust, scalable, and intelligent cybersecurity system that integrates machine learning, deep learning (LSTM), and blockchain to proactively identify and report keylogger activity based on system behavior metrics like CPU usage, memory consumption, keystroke frequency, and network activity.

The system includes:
Real-time monitoring and alert generation
Behavior-based anomaly detection using ML/DL
Tamper-proof audit trail using blockchain
Federated learning with secure model aggregation

🚀 Features
🧠 AI Models: Logistic Regression, Random Forest, and LSTM
🔁 Federated Learning: Models trained locally, aggregated globally
⛓️ Blockchain Logging: Detection events stored immutably using SHA-256
⚡ Real-Time Monitoring: Detect keyloggers instantly using system metrics
📊 Visual Analytics: Graphs, confusion matrices, and ROC curves
🔒 Privacy-Friendly: No raw data leaves local devices

🛠️ Tech Stack
Component	Technology
Programming Language	Python
ML/DL Frameworks	scikit-learn, TensorFlow
Data Scaling	MinMaxScaler, StandardScaler
Blockchain	Custom permissioned blockchain using Python
Visualization	Matplotlib, Seaborn
Real-Time Monitoring	psutil library

🧪 Models & Evaluation
Model	Accuracy	Precision	Recall	F1 Score
Random Forest	89.5%	90%	89%	89%
Logistic Regression	83.5%	84%	82%	83%
LSTM	86.5%	87%	88%	88%

📈 LSTM proved best for temporal patterns; Random Forest was fastest.

🧬 System Architecture
Data Generation
Synthetic data (CPU, Memory, Keystrokes, Network)
Labelled using heuristics
Preprocessing
MinMax scaling
Reshaped into 3D for LSTM
Model Training
Trained locally on multiple devices (federated learning)
Trust scores assigned based on validation accuracy
Blockchain Integration
Stores model updates and detection logs in blocks
Tamper-proof with SHA-256 hashing
Detection & Alerts
Real-time predictions from LSTM
Alert triggered if malicious pattern detected

📂 Sample Code
# Generate and scale data
cpu = psutil.cpu_percent()
memory = psutil.virtual_memory().percent
keystrokes = random.randint(0, 50)
network = random.uniform(0, 1)
label = 1 if keystrokes > 30 and network > 0.6 else 0

# Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, 4)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

📷 **Visual Results**
📊 Accuracy graphs for all models
💡 Feature Importance (RF: Keystroke frequency & network activity dominate)
🔐 Blockchain logs for audit compliance
📉 ROC Curve: LSTM > RF > LR


🔮 Future Scope
Expand to other malware (ransomware, spyware)
Integrate graph-based intrusion detection
Reinforcement learning for adaptive model updates
Deploy in real-world enterprise environments

📚 Authors
N. Keerthi Shivani
T. Sai Sathvika
Sai Chandrasekhar RS
S. Harish

Under the guidance of:
Mrs. B. Spandana, Assistant Professor, Dept. of AIML
Sreyas Institute of Engineering & Technology, JNTUH

📄 License
This project is part of an academic mini project and is licensed under the MIT License for educational purposes.










