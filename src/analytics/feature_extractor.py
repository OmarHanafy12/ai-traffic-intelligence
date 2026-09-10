class FeatureExtractor:

    def extract(self, vehicles, counts, frame_width, frame_height):

        cars = 0
        motorcycles = 0
        buses = 0
        trucks = 0

        for vehicle in vehicles:

            vehicle_type = vehicle.get("class_name", "").lower()

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
            "in_count": counts.get("in_count", 0),
            "out_count": counts.get("out_count", 0),
            "density": density
        }
