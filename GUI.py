import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from model import predict_deepfake  # your AI function

class DeepfakeGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Deepfake Detector GUI")
        self.master.geometry("800x600")

        # Button to upload video
        self.upload_btn = tk.Button(master, text="Upload Video", command=self.upload_video)
        self.upload_btn.pack()

        # Area to show video
        self.video_label = tk.Label(master)
        self.video_label.pack()

        # Status text
        self.status_label = tk.Label(master, text="", font=("Arial", 16))
        self.status_label.pack()

    def upload_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_video(file_path)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # Predict fake/real using your model
                face_img = frame[y:y+h, x:x+w]
                score = predict_deepfake(face_img)
                label = "FAKE" if score > 0.5 else "REAL"
                confidence = score*100 if score > 0.5 else (1-score)*100

                # Show text in GUI
                cv2.putText(frame, f"{label} {confidence:.2f}%", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                self.status_label.config(text=f"Detected: {label} {confidence:.2f}%")

            # Convert frame to Tkinter image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.master.update()

        cap.release()

# Run GUI
root = tk.Tk()
app = DeepfakeGUI(root)
root.mainloop()