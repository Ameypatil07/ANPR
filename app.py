import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import easyocr

# Load the model
detect_fn = tf.saved_model.load("path/to/save/directory")

# Function to perform object detection and OCR
def perform_object_detection_and_ocr(image_path):
    img = cv2.imread(image_path)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # Object detection visualization
    label_map_util = {}  # Assuming you have the necessary code for label_map_util
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + 1,  # Adjust for zero-based indexing
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=0.8,
        agnostic_mode=False
    )

    # OCR
    detection_threshold = 0.7
    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    width, height, _ = image_np.shape

    ocr_results = []

    for idx, box in enumerate(boxes):
        roi = box * np.array([height, width, height, width])
        region = image_np[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]

        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)

        ocr_results.append(ocr_result)

    return image_np_with_detections, ocr_results

# Streamlit app
def main():
    st.title("Automatic Number Plate Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(perform_object_detection_and_ocr(uploaded_file)[0], channels="BGR")

if __name__ == "__main__":
    main()