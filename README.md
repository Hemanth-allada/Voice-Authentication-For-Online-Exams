# 🎙️ Voice Authentication for Online Exams

This project implements a **Voice-Based Authentication System** for secure online examinations.  
It records the student's voice during registration and verifies the voice again during the exam login stage.  
Only if the voice matches, the system grants access — providing enhanced security compared to traditional passwords.

---

## ✅ Features

- 🔐 **Biometric authentication** using voice
- 🧠 Uses **Gaussian Mixture Models (GMM)** for voice matching
- 🎚️ Extracts MFCC features from audio
- 🎧 Real-time **voice recording** and verification
- 💾 Stores trained model securely for later use
- 🚫 Prevents unauthorized access

---

## 🛠️ Technologies Used

| Component | Technology |
|----------|------------|
| Programming Language | Python |
| Libraries | librosa, numpy, sounddevice, soundfile, sklearn |
| Model | Gaussian Mixture Model (GMM) |
| Audio Features | MFCC |

---

## 📂 Project Structure

