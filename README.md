# AI Traffic Intelligence

An end-to-end intelligent traffic analysis and adaptive signal recommendation system.

## Project Structure

```text
ai-traffic-intelligence/
├── data/
│   ├── raw_videos/              # Input traffic videos
│   ├── processed_videos/        # Annotated video outputs
│   └── models/                  # Pretrained weights (e.g., yolov8n.pt, rf_classifier.pkl)
├── src/
│   ├── __init__.py
│   ├── vision/                  # Layer 1: Computer Vision & Object Tracking
│   │   ├── __init__.py
│   │   ├── detector.py          # YOLO inference wrapper (detects vehicles, bounding boxes)
│   │   ├── tracker.py           # Object tracking (ByteTrack/SORT ID management)
│   │   └── counter.py           # Line/Region crossing math & frame overlay drawing
│   ├── analytics/               # Layer 2: Machine Learning & Feature Extraction
│   │   ├── __init__.py
│   │   ├── feature_extractor.py # Aggregates counts, class distributions, density stats
│   │   ├── model.py             # Scikit-Learn classifier (Random Forest/Decision Tree)
│   │   └── recommender.py       # Logic mapping LOW/MEDIUM/HIGH to signal recommendations
│   └── utils/                   # Shared Helper Functions
│       ├── __init__.py
│       └── video_utils.py       # Frame loading, resize utilities, OpenCV video writers
├── app.py                       # Layer 3: Streamlit Interactive UI Application
├── requirements.txt             # Dependency definitions
└── README.md
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd ai-traffic-intelligence
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
