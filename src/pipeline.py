import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision.tracker import VehicleTracker
from vision.Counter import VehicleCounter
from analytics.feature_extractor import FeatureExtractor
from analytics.model import TrafficModel
from analytics.recommender import TrafficRecommender


class TrafficPipeline:
    def __init__(self, model_path=None, line_position=500,
                 frame_width=1280, frame_height=720):

        self.tracker = VehicleTracker(max_age=30)
        self.counter = VehicleCounter(line_y=line_position)

        self.feature_extractor = FeatureExtractor()
        self.model = TrafficModel()
        self.model_ready = False

        self.frame_width = frame_width
        self.frame_height = frame_height

        if model_path and os.path.exists(model_path):
            self.model.load(model_path)
            self.model_ready = True

        self.recommender = TrafficRecommender()

    def process_frame(self, raw_detections, frame_timestamp=None):

        tracked_objects = self.tracker.update(raw_detections)
        current_counts = self.counter.update(tracked_objects)
        extracted_features = self.feature_extractor.extract(
            vehicles=tracked_objects,
            counts=current_counts,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )

        if self.model_ready:
            traffic_status = self.model.predict(extracted_features)
        else:
            traffic_status = "UNKNOWN (no trained model loaded)"

        recommendation = self.recommender.recommend(traffic_status)

        return {
            "tracked_objects": tracked_objects,
            "counts": current_counts,
            "features": extracted_features,
            "traffic_status": traffic_status,
            "recommendation": recommendation,
        }


if __name__ == "__main__":
    print("[+] Initializing Traffic Intelligence Pipeline...")
    pipeline = TrafficPipeline()

    dummy_detections = [
        {"bbox": [100, 200, 150, 250], "confidence": 0.88, "class_name": "car"},
        {"bbox": [300, 400, 380, 480], "confidence": 0.92, "class_name": "truck"},
    ]

    print("[+] Processing dummy frame...")
    result = pipeline.process_frame(dummy_detections)

    print("\n--- Pipeline Execution Summary ---")
    print(f"Tracked Objects Count : {len(result['tracked_objects'])}")
    print(f"Traffic Status        : {result['traffic_status']}")
    print(f"Recommendation        : {result['recommendation']}")
