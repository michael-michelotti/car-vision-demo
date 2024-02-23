import time
import cv2.dnn
from enum import Enum


class AppStatus(Enum):
    SUCCESS = 0
    GENERIC_FAILURE = -1


class CocoObjectClass(Enum):
    PERSON = 0
    CAR = 2


# General configuration parameters
DEBUG_LOGGING = 0
VIDEO_FILENAME = "cars_at_intersection.MOV"
BIT_PER_CHANNEL = 8
YOLO_IMAGE_SIZE = 320
YOLO_SCALE = 1 / ((2 ** BIT_PER_CHANNEL) - 1)
COCO_CLASS_TO_TRACK = CocoObjectClass.CAR
THRESHOLD = 0.5
SUPPRESSION_THRESHOLD = 0.6
OUTPUT_FPS = 5
ASCII_ESC = 27

# Indices for YOLO3 predictions (X, Y, Width, Height, Object Confidence, Class Confidences)
YOLO_PREDICTION_X_POS = 0
YOLO_PREDICTION_Y_POS = 1
YOLO_PREDICTION_W_POS = 2
YOLO_PREDICTION_H_POS = 3
YOLO_PREDICTION_START = 5

# BBOX Configuration
CV2_BLUE = (255, 0, 0)
CV2_GREEN = (0, 255, 0)
CV2_RED = (0, 0, 255)
BOX_COLOR = CV2_BLUE
BOX_STROKE_WIDTH = 2


def get_object_bboxes(model_outputs, coco_class_index):
    """Extract list of prediction boxes and confidence values from YOLO3 model outputs"""
    bbox_locations = []
    confidence_values = []

    for output in model_outputs:
        for prediction in output:
            object_probability = prediction[YOLO_PREDICTION_START + coco_class_index.value]
            if object_probability > THRESHOLD:
                w = int(prediction[YOLO_PREDICTION_W_POS] * YOLO_IMAGE_SIZE)
                h = int(prediction[YOLO_PREDICTION_H_POS] * YOLO_IMAGE_SIZE)
                x = int(prediction[YOLO_PREDICTION_X_POS] * YOLO_IMAGE_SIZE - w / 2)
                y = int(prediction[YOLO_PREDICTION_Y_POS] * YOLO_IMAGE_SIZE - h / 2)
                bbox_locations.append([x, y, w, h])
                confidence_values.append(object_probability)

    indices_to_keep = cv2.dnn.NMSBoxes(bbox_locations, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)
    filtered_bboxes = [bbox_locations[i] for i in indices_to_keep]
    filtered_confidence_values = [confidence_values[i] for i in indices_to_keep]

    return filtered_bboxes, filtered_confidence_values


def add_bboxes_to_frame(image, bboxes, confidence_values):
    """Given a list of boxes and confidence values, draw boxes on given image or frame"""
    height_ratio = image.shape[0] / YOLO_IMAGE_SIZE
    width_ratio = image.shape[1] / YOLO_IMAGE_SIZE
    for i, box in enumerate(bboxes):
        x = int(box[0] * width_ratio)
        y = int(box[1] * height_ratio)
        w = int(box[2] * width_ratio)
        h = int(box[3] * height_ratio)

        cv2.rectangle(image, (x, y), (x+w, y+h), BOX_COLOR, BOX_STROKE_WIDTH)
        class_and_conf_str = f"CAR {confidence_values[i] * 100:.1f}%"
        cv2.putText(image, class_and_conf_str, (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, BOX_COLOR)


def cleanup_resources(capture):
    capture.release()
    cv2.destroyAllWindows()


def main():
    """
    Feed model (pretrained Darknet YOLO3 model) into CV2 DNN implementation,
    draw prediction boxes on frame for COCO_CLASS_TO_TRACK object, stream video at desired OUTPUT_FPS.
    """
    nn = cv2.dnn.readNetFromDarknet(cfgFile="yolov3.cfg", darknetModel="yolov3.weights")
    nn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    nn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    capture = cv2.VideoCapture(VIDEO_FILENAME)
    if not capture.isOpened():
        print("Error opening video capture, exiting...")
        cleanup_resources(capture)
        return AppStatus.GENERIC_FAILURE

    while True:
        # Start frame timer, grab a frame
        frame_start = time.time()
        grabbed, frame = capture.read()
        if not grabbed:
            break

        # Feed frame into YOLO3 model, obtain outputs
        blob = cv2.dnn.blobFromImage(frame, YOLO_SCALE, (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE),
                                     swapRB=True, crop=False)
        nn.setInput(blob)
        layer_names = nn.getLayerNames()
        output_names = [layer_names[i - 1] for i in nn.getUnconnectedOutLayers()]
        outputs = nn.forward(output_names)

        # Feed outputs into utility functions to filter and draw prediction boxes
        car_bboxes, car_confidence_values = get_object_bboxes(outputs, COCO_CLASS_TO_TRACK)
        add_bboxes_to_frame(frame, car_bboxes, car_confidence_values)
        cv2.imshow("Car Vision Video - San Diego Intersection", frame)

        # Wait however long to meet desired OUTPUT_FPS
        processing_time = time.time() - frame_start
        if DEBUG_LOGGING:
            print(f"Frame processed in {processing_time}ms")
        wait_time = max(1, int((1 / OUTPUT_FPS - processing_time) * 1000))

        if cv2.waitKey(wait_time) & 0xFF == ASCII_ESC:
            break

    cleanup_resources(capture)
    return AppStatus.SUCCESS


if __name__ == "__main__":
    main()
