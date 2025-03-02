import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
import cv2
import numpy as np
import json
import tensorflow as tf

# Google Cloud Project and Pub/Sub topics
PROJECT_ID = "pure-spring-449721-p3"
INPUT_TOPIC = f"projects/{PROJECT_ID}/topics/input-topic"
OUTPUT_TOPIC = f"projects/{PROJECT_ID}/topics/output-topic"

# Google Cloud Storage Bucket
BUCKET_NAME = "pure-spring-449721-p3-bucket"

# Load Pretrained Models for Pedestrian Detection and Depth Estimation
pedestrian_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
depth_model = tf.keras.models.load_model("depth_estimation_model.h5")  # Replace with your model

class ProcessFrame(beam.DoFn):
    def process(self, element):
        """Detects pedestrians and estimates depth."""
        try:
            # Decode Pub/Sub message
            message = json.loads(element.decode("utf-8"))
            image_bytes = np.frombuffer(message["image"], dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Pedestrian detection
            pedestrians = pedestrian_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 100))
            
            results = []
            for (x, y, w, h) in pedestrians:
                roi = image[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, (128, 128)) / 255.0
                roi_expanded = np.expand_dims(roi_resized, axis=0)
                
                # Predict depth using model
                depth = depth_model.predict(roi_expanded)[0][0]
                
                results.append({
                    "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "estimated_depth": float(depth)
                })
            
            if results:
                output_data = json.dumps({"detections": results})
                yield output_data

        except Exception as e:
            print(f"Error processing frame: {e}")

def run():
    """Sets up and runs the Apache Beam pipeline."""
    options = PipelineOptions(
        streaming=True,
        project=PROJECT_ID,
        region="northamerica-northeast2",
        temp_location=f"gs://{BUCKET_NAME}/temp",
        staging_location=f"gs://{BUCKET_NAME}/staging",
    )
    options.view_as(StandardOptions).runner = "DataflowRunner"

    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Read from Pub/Sub" >> beam.io.ReadFromPubSub(topic=INPUT_TOPIC).with_output_types(bytes)
            | "Detect Pedestrians & Estimate Depth" >> beam.ParDo(ProcessFrame())
            | "Write to Pub/Sub" >> beam.io.WriteToPubSub(topic=OUTPUT_TOPIC)
        )

if __name__ == "__main__":
    run()
