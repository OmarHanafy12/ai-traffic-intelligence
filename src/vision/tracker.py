class VehicleTracker:

    def __init__(self, iou_threshold=0.3, max_age=30):

        self.iou_threshold = iou_threshold
        self.max_age = max_age

        self.next_id = 1
        self.tracks = {}

    def iou(self, box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

        union = area1 + area2 - intersection

        if union == 0:
            return 0

        return intersection / union

    def update(self, detections):

        tracked = []

        matched_ids = set()

        # Increase age of all existing tracks
        for track_id in self.tracks:
            self.tracks[track_id]["age"] += 1

        for detection in detections:

            best_id = None
            best_iou = 0

            for track_id, track_data in self.tracks.items():

                if track_id in matched_ids:
                    continue

                score = self.iou(
                    detection["bbox"],
                    track_data["detection"]["bbox"]
                )

                if score > best_iou:
                    best_iou = score
                    best_id = track_id

            # Existing vehicle
            if best_id is not None and best_iou >= self.iou_threshold:

                vehicle_id = best_id

            # New vehicle
            else:

                vehicle_id = self.next_id
                self.next_id += 1

            previous_detection = self.tracks.get(vehicle_id)

            previous_y = None

            if previous_detection:

                previous_y = previous_detection["detection"].get(
                    "center",
                    (
                        (previous_detection["detection"]["bbox"][0] +
                         previous_detection["detection"]["bbox"][2]) / 2,

                        (previous_detection["detection"]["bbox"][1] +
                         previous_detection["detection"]["bbox"][3]) / 2
                    )
                )[1]

            x1, y1, x2, y2 = detection["bbox"]

            center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )

            vehicle = detection.copy()

            vehicle["id"] = vehicle_id
            vehicle["center"] = center

            if previous_y is not None:
                vehicle["previous_y"] = previous_y

            self.tracks[vehicle_id] = {
                "detection": vehicle.copy(),
                "age": 0
            }

            matched_ids.add(vehicle_id)

            tracked.append(vehicle)

        # Remove old tracks
        expired_ids = [
            track_id
            for track_id, track_data in self.tracks.items()
            if track_data["age"] > self.max_age
        ]

        for track_id in expired_ids:
            del self.tracks[track_id]

        return tracked
