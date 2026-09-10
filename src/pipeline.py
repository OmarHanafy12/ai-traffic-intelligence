import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision.Tracker import Tracker
from vision.Counter import Counter
from analysis.feature_extractor import FeatureExtractor
from analysis.model import TrafficModel
from analysis.Recommender import TrafficRecommender


class TrafficPipeline:
    def __init__(self, model_path=None, line_position=500):
       
        self.tracker = Tracker(max_age=30)
        self.counter = Counter(line_y=line_position)
        
        self.feature_extractor = FeatureExtractor()
        self.model = TrafficModel()
        
        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path)
            
        self.recommender = TrafficRecommender()

    def process_frame(self, raw_detections, frame_timestamp=None):
       
        tracked_objects = self.tracker.update(raw_detections)
        current_counts = self.counter.update(tracked_objects)
        extracted_features = self.feature_extractor.extract(
            counts=current_counts, 
            tracked_objects=tracked_objects,
            timestamp=frame_timestamp
        )
        traffic_status = self.model.predict(extracted_features)
        recommendation = self.recommender.get_recommendation(traffic_status)
        
      return {
            "tracked_objects": tracked_objects,
            "counts": current_counts,
            "features": extracted_features,
            "traffic_status": traffic_status,
            "recommendation": recommendation
        }

if __name__ == "__main__":
    print("[+] Initializing Traffic Intelligence Pipeline...")
    pipeline = TrafficPipeline()

    dummy_detections = [
        [100, 200, 150, 250, 0.88, "car"],
        [300, 400, 380, 480, 0.92, "truck"]
    ]

    print("[+] Processing dummy frame...")
    result = pipeline.process_frame(dummy_detections)

    print("\n--- Pipeline Execution Summary ---")
    print(f"Tracked Objects Count : {len(result['tracked_objects'])}")
    print(f"Traffic Status        : {result['traffic_status']}")
    print(f"Recommendation        : {result['recommendation']}")
