from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
import re

# ===========================
# FLASK APP
# ===========================
app = Flask(__name__)  # No static folder needed

# ===========================
# LOAD MODEL AND ENCODER
# ===========================
MODEL_PATH = "gesture_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
expected_features = model.n_features_in_

# ===========================
# MEDIA PIPE
# ===========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,  # use False for live webcam detection
    max_num_hands=2,
    min_detection_confidence=0.7
)

# ===========================
# ROUTES
# ===========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = data['image']

        # Convert base64 image to numpy array
        img_str = re.search(r'base64,(.*)', img_data).group(1)
        img_bytes = base64.b64decode(img_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process image with MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            return jsonify({'gesture': 'No hand detected', 'confidence': 0})

        hand_landmarks = result.multi_hand_landmarks[0]
        features = [coord for lm in hand_landmarks.landmark for coord in [lm.x, lm.y, lm.z]]

        # Adjust feature length
        if len(features) < expected_features:
            features += [0] * (expected_features - len(features))
        elif len(features) > expected_features:
            features = features[:expected_features]

        # Predict gesture
        prediction = model.predict([features])
        gesture_name = encoder.inverse_transform(prediction)[0]

        return jsonify({'gesture': gesture_name, 'confidence': 100})

    except Exception as e:
        return jsonify({'gesture': 'Error', 'confidence': 0, 'message': str(e)})

# ===========================
# RUN APP
# ===========================
if __name__ == '__main__':
    app.run(debug=True)
#http://127.0.0.1:5000/