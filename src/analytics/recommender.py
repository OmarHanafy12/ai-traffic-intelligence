# recommender
class TrafficRecommender:
    def recommend(self, traffic_level):
        recommendations = {
            "LOW":
                "Keep normal signal timing.",
            "MEDIUM":
                "Increase green-light duration moderately.",
            "HIGH":
                "Increase green-light duration significantly."
        }
        return recommendations.get(
            traffic_level,
            "Unknown traffic level."
        )
