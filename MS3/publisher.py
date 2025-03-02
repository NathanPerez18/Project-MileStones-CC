from google.cloud import pubsub_v1
import cv2
import json
import time

publisher = pubsub_v1.PublisherClient()
topic_path = f"projects/pure-spring-449721-p3/topics/input-topic"

cap = cv2.VideoCapture(0)  # Use webcam or video file

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    _, img_encoded = cv2.imencode(".jpg", frame)
    img_bytes = img_encoded.tobytes()

    message = json.dumps({"image": img_bytes.decode("latin1")}).encode("utf-8")
    publisher.publish(topic_path, message)
    time.sleep(0.5)  # Simulate real-time streaming

cap.release()
