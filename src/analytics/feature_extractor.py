# feature_extractor

class FeatureExtractor:

    def extract(self, vehicles, counts, frame_width, frame_height):

        cars = 0
        motorcycles = 0
        buses = 0
        trucks = 0

        for vehicle in vehicles:

            vehicle_type = vehicle["class_name"]

            if vehicle_type == "car":
                cars += 1

            elif vehicle_type == "motorcycle":
                motorcycles += 1

            elif vehicle_type == "bus":
                buses += 1

            elif vehicle_type == "truck":
                trucks += 1

        total = len(vehicles)

        area = frame_width * frame_height

        density = total / area if area else 0

        return {
            "vehicle_count": total,
            "cars": cars,
            "motorcycles": motorcycles,
            "buses": buses,
            "trucks": trucks,
            "in_count": counts["in_count"],
            "out_count": counts["out_count"],
            "density": density
        }
