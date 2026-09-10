class VehicleCounter:

    def __init__(self, line_y):

        self.line_y = line_y

        self.counted = set()

        self.in_count = 0
        self.out_count = 0

    def update(self, vehicles):

        for vehicle in vehicles:

            vehicle_id = vehicle["id"]

            current_y = vehicle["center"][1]

            previous_y = vehicle.get(
                "previous_y",
                current_y
            )

            # Moving downward
            if previous_y < self.line_y <= current_y:

                if vehicle_id not in self.counted:

                    self.in_count += 1
                    self.counted.add(vehicle_id)

            # Moving upward
            elif previous_y > self.line_y >= current_y:

                if vehicle_id not in self.counted:

                    self.out_count += 1
                    self.counted.add(vehicle_id)

        return self.get_counts()

    def get_counts(self):

        return {
            "in_count": self.in_count,
            "out_count": self.out_count,
            "total_count": self.in_count + self.out_count
        }
