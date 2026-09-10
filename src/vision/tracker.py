class VehicleTracker:

    def __init__(self, iou_threshold=0.3, max_age=5):
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

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union if union else 0

    def update(self, detections):

        tracked = []
        updated_track_ids = set()

        for detection in detections:

            best_id = None
            best_iou = 0

            for track_id, track_data in self.tracks.items():

                score = self.iou(
                    detection["bbox"],
                    track_data["detection"]["bbox"]
                )

                if score > best_iou:
                    best_iou = score
                    best_id = track_id

            if best_iou >= self.iou_threshold:
                vehicle_id = best_id
            else:
                vehicle_id = self.next_id
                self.next_id += 1


            self.tracks[vehicle_id] = {
                "detection": detection,
                "age": 0
            }
            updated_track_ids.add(vehicle_id)

            vehicle = detection.copy()
            vehicle["id"] = vehicle_id

            x1, y1, x2, y2 = detection["bbox"]

            vehicle["center"] = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )

            tracked.append(vehicle)

            unseen_ids = set(self.tracks.keys()) - updated_track_ids
        for track_id in unseen_ids:
            self.tracks[track_id]["age"] += 1
            if self.tracks[track_id]["age"] > self.max_age:
                del self.tracks[track_id]

        return tracked

