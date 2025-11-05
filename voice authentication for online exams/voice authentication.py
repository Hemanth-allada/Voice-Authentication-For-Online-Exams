"""
Voice Authentication System for Online Exams
This system registers student voices during exam registration and verifies them during the exam.
"""

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import pickle
import os
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class VoiceAuthenticator:
    """Voice authentication system using GMM (Gaussian Mixture Models)"""
    
    def __init__(self, models_dir="voice_models"):
        self.models_dir = models_dir
        self.sample_rate = 16000
        self.duration = 5  # seconds
        
        # Create directory for storing voice models
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def extract_features(self, audio_path):
        """Extract MFCC features from audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Combine features
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            features = features.T  # Transpose to get (time, features) shape
            
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def record_audio(self, filename, duration=None):
        """Record audio from microphone"""
        if duration is None:
            duration = self.duration
            
        print(f"Recording for {duration} seconds... Please speak clearly.")
        print("Say: 'My name is [Your Name] and I am taking this exam today.'")
        
        # Record audio
        recording = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        
        # Save to file
        write(filename, self.sample_rate, recording)
        print(f"Recording saved to {filename}")
        
        return filename
    
    def register_student(self, student_id, num_samples=3):
        """Register a student's voice by recording multiple samples"""
        print(f"\n=== Voice Registration for Student ID: {student_id} ===")
        print("You will be asked to record your voice 3 times.")
        print("Please speak the same phrase each time for consistency.\n")
        
        all_features = []
        
        for i in range(num_samples):
            print(f"\nRecording sample {i+1}/{num_samples}")
            input("Press Enter when ready to record...")
            
            # Record audio
            temp_file = f"temp_registration_{student_id}_{i}.wav"
            self.record_audio(temp_file)
            
            # Extract features
            features = self.extract_features(temp_file)
            if features is not None:
                all_features.append(features)
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        if len(all_features) == 0:
            print("Error: Could not extract features from recordings.")
            return False
        
        # Combine all features
        combined_features = np.vstack(all_features)
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(combined_features)
        
        # Train GMM model
        print("\nTraining voice model...")
        gmm = GaussianMixture(n_components=16, covariance_type='diag', 
                             max_iter=200, random_state=42)
        gmm.fit(normalized_features)
        
        # Save model and scaler
        model_data = {
            'gmm': gmm,
            'scaler': scaler,
            'student_id': student_id,
            'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        model_path = os.path.join(self.models_dir, f"{student_id}_voice_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Voice model successfully created for Student ID: {student_id}")
        print(f"Model saved at: {model_path}")
        return True
    
    def verify_student(self, student_id, threshold=-50):
        """Verify a student's identity during exam"""
        print(f"\n=== Voice Verification for Student ID: {student_id} ===")
        
        # Load the student's voice model
        model_path = os.path.join(self.models_dir, f"{student_id}_voice_model.pkl")
        
        if not os.path.exists(model_path):
            print(f"Error: No voice model found for Student ID: {student_id}")
            print("Please register first.")
            return False
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        gmm = model_data['gmm']
        scaler = model_data['scaler']
        
        print("\nPlease speak for verification...")
        input("Press Enter when ready to record...")
        
        # Record verification audio
        temp_file = f"temp_verification_{student_id}.wav"
        self.record_audio(temp_file)
        
        # Extract features
        features = self.extract_features(temp_file)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if features is None:
            print("Error: Could not extract features from recording.")
            return False
        
        # Normalize features using saved scaler
        normalized_features = scaler.transform(features)
        
        # Calculate likelihood score
        score = gmm.score(normalized_features)
        
        print(f"\nVerification Score: {score:.2f}")
        print(f"Threshold: {threshold}")
        
        # Verify against threshold
        if score > threshold:
            print(f"\n✓ VERIFICATION SUCCESSFUL!")
            print(f"Student ID {student_id} is authenticated.")
            return True
        else:
            print(f"\n✗ VERIFICATION FAILED!")
            print(f"Voice does not match registered profile for Student ID {student_id}.")
            return False
    
    def continuous_verification(self, student_id, num_checks=3, interval=300):
        """Perform continuous verification during exam at intervals"""
        print(f"\n=== Continuous Verification Mode ===")
        print(f"Verification will be performed {num_checks} times during the exam.")
        
        passed_checks = 0
        
        for i in range(num_checks):
            print(f"\n--- Verification Check {i+1}/{num_checks} ---")
            
            if i > 0:
                print(f"Waiting for next verification checkpoint...")
                # In real scenario, you'd wait for 'interval' seconds
                input("Press Enter to continue to next verification...")
            
            result = self.verify_student(student_id)
            
            if result:
                passed_checks += 1
            else:
                print("\n⚠ Warning: Verification failed. Exam may be flagged.")
        
        print(f"\n=== Verification Summary ===")
        print(f"Passed: {passed_checks}/{num_checks} checks")
        
        if passed_checks >= num_checks * 0.7:  # 70% threshold
            print("✓ Overall verification: PASSED")
            return True
        else:
            print("✗ Overall verification: FAILED")
            return False


class ExamSystem:
    """Main exam system integrating voice authentication"""
    
    def __init__(self):
        self.voice_auth = VoiceAuthenticator()
        self.registered_students = set()
    
    def register_for_exam(self, student_id):
        """Register student for exam with voice authentication"""
        print("\n" + "="*60)
        print("EXAM REGISTRATION WITH VOICE AUTHENTICATION")
        print("="*60)
        
        success = self.voice_auth.register_student(student_id)
        
        if success:
            self.registered_students.add(student_id)
            print(f"\n✓ Student {student_id} successfully registered for exam!")
        else:
            print(f"\n✗ Registration failed for Student {student_id}")
        
        return success
    
    def start_exam(self, student_id):
        """Start exam with voice verification"""
        print("\n" + "="*60)
        print("ONLINE EXAM - VOICE VERIFICATION REQUIRED")
        print("="*60)
        
        # Verify student before starting exam
        verified = self.voice_auth.verify_student(student_id)
        
        if verified:
            print("\n✓ You may now begin the exam.")
            print("\nNote: Periodic voice verification will be required during the exam.")
            return True
        else:
            print("\n✗ Access denied. Verification failed.")
            return False
    
    def conduct_exam_with_monitoring(self, student_id):
        """Conduct exam with continuous voice monitoring"""
        print("\n" + "="*60)
        print("EXAM IN PROGRESS - VOICE MONITORING ACTIVE")
        print("="*60)
        
        # Initial verification
        if not self.start_exam(student_id):
            return False
        
        # Simulate exam duration with periodic checks
        print("\n[Exam started - You can now answer questions]")
        print("\nYou will be prompted for voice verification at intervals.")
        
        # Perform continuous verification
        result = self.voice_auth.continuous_verification(student_id, num_checks=2)
        
        if result:
            print("\n✓ Exam completed successfully with valid authentication.")
        else:
            print("\n✗ Exam flagged due to authentication issues.")
        
        return result


def main():
    """Main function to demonstrate the voice authentication system"""
    
    print("\n" + "="*60)
    print("VOICE AUTHENTICATION SYSTEM FOR ONLINE EXAMS")
    print("="*60)
    
    exam_system = ExamSystem()
    
    while True:
        print("\n" + "-"*60)
        print("MAIN MENU")
        print("-"*60)
        print("1. Register for Exam (Voice Registration)")
        print("2. Start Exam (Voice Verification)")
        print("3. Full Exam with Monitoring")
        print("4. Exit")
        print("-"*60)
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            student_id = input("\nEnter Student ID: ").strip()
            exam_system.register_for_exam(student_id)
        
        elif choice == '2':
            student_id = input("\nEnter Student ID: ").strip()
            exam_system.start_exam(student_id)
        
        elif choice == '3':
            student_id = input("\nEnter Student ID: ").strip()
            exam_system.conduct_exam_with_monitoring(student_id)
        
        elif choice == '4':
            print("\nThank you for using the Voice Authentication System!")
            break
        
        else:
            print("\n✗ Invalid choice. Please try again.")


if __name__ == "__main__":
    print("\nInstalling required packages...")
    print("Run: pip install numpy librosa sounddevice soundfile scipy scikit-learn")
    print("\nStarting system...\n")
    
    main()