import os

from vision.tracker import VehicleTracker
from vision.Counter import VehicleCounter

from analytics.feature_extractor import FeatureExtractor
from analytics.model import TrafficModel
from analytics.recommender import TrafficRecommender


class TrafficPipeline:

    def __init__(
        self,
        model_path=None,
        line_position=500,
        frame_width=1280,
        frame_height=720
    ):

        # -------------------------
        # Configuration
        # -------------------------

        self.frame_width = frame_width
        self.frame_height = frame_height

        # -------------------------
        # Vision
        # -------------------------

        self.tracker = VehicleTracker(
            iou_threshold=0.3,
            max_age=30
        )

        self.counter = VehicleCounter(
            line_y=line_position
        )

        # -------------------------
        # Analytics / Machine Learning
        # -------------------------

        self.feature_extractor = FeatureExtractor()

        self.model = TrafficModel()

        # Load trained model if available
        if model_path and os.path.exists(model_path):

            print(f"[+] Loading model: {model_path}")

            self.model.load(model_path)

        else:

            print("[!] No trained model loaded.")

        # -------------------------
        # Recommendation
        # -------------------------

        self.recommender = TrafficRecommender()

    def process_frame(
        self,
        raw_detections,
        frame_width=None,
        frame_height=None,
        frame_timestamp=None
    ):

        # Use default dimensions if not provided
        if frame_width is None:
            frame_width = self.frame_width

        if frame_height is None:
            frame_height = self.frame_height

        # -------------------------
        # 1. Track vehicles
        # -------------------------

        tracked_objects = self.tracker.update(
            raw_detections
        )

        # -------------------------
        # 2. Count vehicles
        # -------------------------

        current_counts = self.counter.update(
            tracked_objects
        )

        # -------------------------
        # 3. Extract features
        # -------------------------

        extracted_features = self.feature_extractor.extract(
            vehicles=tracked_objects,
            counts=current_counts,
            frame_width=frame_width,
            frame_height=frame_height
        )

        # -------------------------
        # 4. Predict traffic level
        # -------------------------

        traffic_status = None

        try:

            traffic_status = self.model.predict(
                extracted_features
            )

        except Exception as e:

            print(
                f"[!] Traffic model prediction skipped: {e}"
            )

        # -------------------------
        # 5. Generate recommendation
        # -------------------------

        recommendation = self.recommender.recommend(
            traffic_status
        ) if traffic_status else "No recommendation available."

        # -------------------------
        # Final result
        # -------------------------

        return {

            "tracked_objects": tracked_objects,

            "counts": current_counts,

            "features": extracted_features,

            "traffic_status": traffic_status,

            "recommendation": recommendation,

            "timestamp": frame_timestamp
        }


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    print(
        "[+] Initializing Traffic Intelligence Pipeline..."
    )

    pipeline = TrafficPipeline()

    # Correct detection format
    dummy_detections = [

        {
            "bbox": [100, 200, 150, 250],
            "confidence": 0.88,
            "class_name": "car"
        },

        {
            "bbox": [300, 400, 380, 480],
            "confidence": 0.92,
            "class_name": "truck"
        }

    ]

    print(
        "[+] Processing dummy frame..."
    )

    result = pipeline.process_frame(
        raw_detections=dummy_detections,
        frame_width=1280,
        frame_height=720
    )

    print("\n--- Pipeline Execution Summary ---")

    print(
        f"Tracked Objects Count : "
        f"{len(result['tracked_objects'])}"
    )

    print(
        f"Vehicle Counts        : "
        f"{result['counts']}"
    )

    print(
        f"Features              : "
        f"{result['features']}"
    )

    print(
        f"Traffic Status        : "
        f"{result['traffic_status']}"
    )

    print(
        f"Recommendation        : "
        f"{result['recommendation']}"
    )
