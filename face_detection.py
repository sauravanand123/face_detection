import cv2
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import Button, Label, filedialog, messagebox
from PIL import Image, ImageTk

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.vdo = cv2.VideoCapture(0)
        self.model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.modellogistic = LogisticRegression()
        self.is_running = False
        self.x_train, self.y_train = [], []
        
        # UI Elements
        self.label = Label(root)
        self.label.pack()
        
        self.btn_capture_with_beard = Button(root, text="Capture Beard", command=self.capture_beard)
        self.btn_capture_with_beard.pack(pady=10)
        
        self.btn_capture_no_beard = Button(root, text="Capture No Beard", command=self.capture_no_beard)
        self.btn_capture_no_beard.pack(pady=10)
        
        self.btn_train = Button(root, text="Train Model", command=self.train_model)
        self.btn_train.pack(pady=10)
        
        self.btn_detect = Button(root, text="Real-Time Detection", command=self.start_detection)
        self.btn_detect.pack(pady=10)
        
        self.btn_quit = Button(root, text="Quit", command=self.quit_app)
        self.btn_quit.pack(pady=10)
    
    def capture_image(self, label):
        """Capture image and save it with the specified label."""
        ret, frame = self.vdo.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            return
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90))
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (120, 120))
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB) / 255.0
            sample = gray_face.flatten()
            self.x_train.append(sample)
            self.y_train.append(label)
            messagebox.showinfo("Success", f"Captured {label} image")
    
    def capture_beard(self):
        self.capture_image('beard')
    
    def capture_no_beard(self):
        self.capture_image('no beard')
    
    def train_model(self):
        """Train the Logistic Regression model."""
        if not self.x_train or not self.y_train:
            messagebox.showerror("Error", "No training data available")
            return
        
        self.modellogistic.fit(self.x_train, self.y_train)
        messagebox.showinfo("Success", "Model trained successfully")
    
    def start_detection(self):
        """Start real-time face detection and classification."""
        if not hasattr(self.modellogistic, "coef_"):
            messagebox.showerror("Error", "Model not trained")
            return
        
        self.is_running = True
        self.detect_faces()
    
    def detect_faces(self):
        """Real-time face detection and classification."""
        if not self.is_running:
            return
        
        ret, frame = self.vdo.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90))
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (120, 120))
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB) / 255.0
            sample = gray_face.flatten().reshape(1, -1)
            
            # Make prediction
            prediction = self.modellogistic.predict(sample)
            label = prediction[0]
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        # Convert frame to Tkinter-compatible format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        
        if self.is_running:
            self.root.after(10, self.detect_faces)

    def quit_app(self):
        """Quit the application."""
        self.is_running = False
        self.vdo.release()
        self.root.destroy()

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
