import cv2
import numpy as np
import argparse
import time
import threading
import pygame
from imutils.video import VideoStream
import os

class ImprovedDrowsinessDetector:
    def __init__(self, args):
        # Parameters
        self.EYE_RATIO_THRESHOLD = 0.25  # Eye height/width ratio threshold
        self.EYE_CLOSED_FRAMES = 10
        self.YAWN_RATIO_THRESHOLD = 0.6  # Mouth height/width ratio threshold
        self.YAWN_FRAMES = 10
        self.FACE_MISSING_FRAMES = 5  # Number of consecutive frames before face missing alert
        self.eye_counter = 0
        self.yawn_counter = 0
        self.face_missing_counter = 0  # New counter for tracking missing face
        self.alarm_status = False
        self.yawn_status = False
        self.face_missing_status = False  # New status for face detection
        self.alarm_path = args["alarm"]
        
        # Initialize pygame for better audio handling
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound(self.alarm_path)
        
        # Start video stream
        print("[INFO] Starting video stream...")
        self.vs = VideoStream(src=args["webcam"]).start()
        time.sleep(1.0)
        
        # Load OpenCV face detector
        print("[INFO] Loading face detector...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise ValueError("Error loading face cascade. Check if file exists.")
        
        # Load OpenCV eye detector
        print("[INFO] Loading eye detector...")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        if self.eye_cascade.empty():
            raise ValueError("Error loading eye cascade. Check if file exists.")
        
        # Load smile detector directly (skip the mouth detector that was failing)
        print("[INFO] Loading smile detector for mouth detection...")
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        if self.mouth_cascade.empty():
            print("[WARNING] Smile detector not found!")
            # Create a dummy cascade to prevent errors
            self.mouth_cascade = None

    def sound_alarm(self):
        """Play alarm sound in a separate thread"""
        try:
            self.alarm_sound.play()
        except Exception as e:
            print(f"Error playing alarm sound: {e}")

    def calculate_aspect_ratio(self, region):
        """Calculate aspect ratio (height/width) of a region"""
        h, w = region.shape[:2]
        return float(h) / w if w > 0 else 0

    def process_frame(self, frame):
        """Process each frame for drowsiness detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to improve contrast
            gray = cv2.equalizeHist(gray)
            
            # Detect faces with improved parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # If no face is detected, increment missing face counter and reset other counters
            if len(faces) == 0:
                self.face_missing_counter += 1
                # Don't reset drowsiness counters here to preserve their state
                
                cv2.putText(frame, f"No face detected ({self.face_missing_counter}/{self.FACE_MISSING_FRAMES})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Check if face has been missing for too long
                if self.face_missing_counter >= self.FACE_MISSING_FRAMES:
                    if not self.face_missing_status:
                        self.face_missing_status = True
                        threading.Thread(target=self.sound_alarm).start()
                    
                    cv2.putText(frame, "FACE MISSING ALERT!", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Still display drowsiness and yawn counters even when face is not detected
                cv2.putText(frame, f"Drowsy Frames: {self.eye_counter}/{self.EYE_CLOSED_FRAMES}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                cv2.putText(frame, f"Yawn Frames: {self.yawn_counter}/{self.YAWN_FRAMES}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                return frame
            else:
                # Only reset the face missing counter when face is detected
                was_missing = self.face_missing_status  # Store previous state
                self.face_missing_counter = 0
                self.face_missing_status = False
                
                # Don't trigger alarm when face is detected again
                if was_missing:
                    # Just reset the status without playing alarm
                    pass
            
            # Process the first face detected (assume driver's face)
            (x, y, w, h) = faces[0]
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region for eye and mouth detection
            face_gray = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]
            
            # Detect eyes in the face region with improved parameters
            eyes = []
            if self.eye_cascade is not None:
                try:
                    eyes = self.eye_cascade.detectMultiScale(
                        face_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(25, 25),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                except Exception as e:
                    print(f"Error detecting eyes: {e}")
            
            # Variables for eye analysis
            left_eye = None
            right_eye = None
            eye_ratios = []
            
            # Process detected eyes
            for (ex, ey, ew, eh) in eyes:
                # Draw rectangle around the eye
                cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                
                # Extract eye region
                eye_region = face_gray[ey:ey+eh, ex:ex+ew]
                
                # Apply thresholding to help determine if eye is open or closed
                _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)
                
                # Count white pixels (eyeball area) in thresholded image
                white_pixels = cv2.countNonZero(threshold_eye)
                total_pixels = threshold_eye.size
                white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
                
                # Calculate aspect ratio of eye
                eye_ratio = self.calculate_aspect_ratio(eye_region)
                eye_ratios.append(eye_ratio)
                
                # Store eye based on position (left/right side of face)
                if ex < w/2:  # Left half of face
                    left_eye = {'ratio': eye_ratio, 'white_ratio': white_ratio, 'region': (ex, ey, ew, eh)}
                else:  # Right half of face
                    right_eye = {'ratio': eye_ratio, 'white_ratio': white_ratio, 'region': (ex, ey, ew, eh)}
            
            # Check for eye closure based on height/width ratio and white pixel ratio
            eyes_closed = False
            
            # If we have aspect ratios for eyes
            if eye_ratios:
                avg_eye_ratio = sum(eye_ratios) / len(eye_ratios)
                
                # If average ratio is below threshold, eyes are likely closed
                if avg_eye_ratio < self.EYE_RATIO_THRESHOLD:
                    eyes_closed = True
                    self.eye_counter += 1
                else:
                    self.eye_counter = max(0, self.eye_counter - 1)  # Decrement counter but don't go below 0
            else:
                # No eyes detected - this is important for drowsiness detection
                self.eye_counter += 1
                eyes_closed = True
            
            # Handle drowsiness alert
            if self.eye_counter >= self.EYE_CLOSED_FRAMES:
                if not self.alarm_status:
                    self.alarm_status = True
                    threading.Thread(target=self.sound_alarm).start()
                
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.alarm_status = False
            
            # Set the region of interest for mouth detection (lower half of face)
            mouth_y = y + int(h/2)
            mouth_h = h - int(h/2)
            mouth_roi_gray = gray[mouth_y:mouth_y+mouth_h, x:x+w]
            mouth_roi_color = frame[mouth_y:mouth_y+mouth_h, x:x+w]
            
            # Detect mouth with improved parameters (using smile detector)
            mouths = []
            if self.mouth_cascade is not None:
                try:
                    mouths = self.mouth_cascade.detectMultiScale(
                        mouth_roi_gray,
                        scaleFactor=1.7,  # Increased scale factor for better performance
                        minNeighbors=22,  # Increased to reduce false positives
                        minSize=(25, 15),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                except Exception as e:
                    print(f"Error detecting mouth: {e}")
            
            # Check for yawning (large mouth detected with specific ratio)
            yawning = False
            for (mx, my, mw, mh) in mouths:
                # Draw mouth rectangle
                cv2.rectangle(mouth_roi_color, (mx, my), (mx+mw, my+mh), (0, 255, 255), 2)
                
                # Extract mouth region
                mouth_region = mouth_roi_gray[my:my+mh, mx:mx+mw]
                
                # Calculate mouth aspect ratio (height/width)
                mouth_ratio = self.calculate_aspect_ratio(mouth_region)
                
                # Display mouth ratio
                cv2.putText(frame, f"Mouth Ratio: {mouth_ratio:.2f}", (x, y+h+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # If mouth is relatively tall compared to width, consider it a yawn
                if mouth_ratio > self.YAWN_RATIO_THRESHOLD and mh > 15:
                    yawning = True
            
            if yawning:
                self.yawn_counter += 1
                
                # Trigger yawn alert after a few consistent frames
                if self.yawn_counter >= self.YAWN_FRAMES:
                    if not self.yawn_status:
                        self.yawn_status = True
                        threading.Thread(target=self.sound_alarm).start()
                    
                    cv2.putText(frame, "YAWN ALERT!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.yawn_counter = max(0, self.yawn_counter - 1)  # Decrement counter but don't go below 0
                self.yawn_status = False
            
            # Display eye aspect ratio when eyes are detected
            if eye_ratios:
                cv2.putText(frame, f"Eye Ratio: {avg_eye_ratio:.2f}", (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Display eye and mouth detection status
            eye_status = f"Eyes: {'Closed' if eyes_closed else 'Open'}"
            mouth_status = f"Mouth: {'Yawning' if yawning else 'Normal'}"
            
            cv2.putText(frame, eye_status, (10, frame.shape[0]-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, mouth_status, (10, frame.shape[0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display drowsiness counter
            cv2.putText(frame, f"Drowsy Frames: {self.eye_counter}/{self.EYE_CLOSED_FRAMES}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Display yawn counter
            cv2.putText(frame, f"Yawn Frames: {self.yawn_counter}/{self.YAWN_FRAMES}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Display face detection status
            cv2.putText(frame, f"Face Detected", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return frame
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame

    def run(self):
        """Main loop for processing video frames"""
        try:
            while True:
                # Read frame from video stream
                frame = self.vs.read()
                
                # Skip if frame is None
                if frame is None:
                    print("Warning: Received empty frame")
                    time.sleep(0.1)
                    continue
                
                # Process frame for drowsiness detection
                processed_frame = self.process_frame(frame)
                
                # Display the processed frame
                cv2.imshow("Drowsiness Detection", processed_frame)
                
                # Break loop on 'q' key press (using a shorter wait time)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        except KeyboardInterrupt:
            print("\n[INFO] Detected keyboard interrupt, cleaning up...")
        except Exception as e:
            print(f"Error in run method: {e}")
        finally:
            # Clean up
            print("[INFO] Cleaning up...")
            cv2.destroyAllWindows()
            self.vs.stop()
            pygame.mixer.quit()

def main():
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0,
                    help="index of webcam on system")
    ap.add_argument("-a", "--alarm", type=str, default="alert.wav",
                    help="path to alarm sound file")
    args = vars(ap.parse_args())
    
    # Check if alarm file exists
    if not os.path.exists(args["alarm"]):
        print(f"[WARNING] Alarm file '{args['alarm']}' not found. Using fallback.")
        # Find a default Windows sound file as fallback
        if os.path.exists("C:\\Windows\\Media\\alarm01.wav"):
            args["alarm"] = "C:\\Windows\\Media\\alarm01.wav"
    
    try:
        # Create and run the drowsiness detector
        detector = ImprovedDrowsinessDetector(args)
        detector.run()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()