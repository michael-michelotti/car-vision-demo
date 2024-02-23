import cv2.dnn
from enum import Enum

YOLO_IMAGE_SIZE = 320
BIT_PER_CHANNEL = 8
YOLO_SCALE = 1 / ((2 ** BIT_PER_CHANNEL) - 1)
COCO_CLASSES_CAR_INDEX = 2
COCO_CLASSES_PERSON_INDEX = 0

THRESHOLD = 0.5
SUPPRESSION_THRESHOLD = 0.6

YOLO_PREDICTION_X_POS = 0
YOLO_PREDICTION_Y_POS = 1
YOLO_PREDICTION_W_POS = 2
YOLO_PREDICTION_H_POS = 3
YOLO_PREDICTION_START = 5
CV2_BLUE = (255, 0, 0)
BOX_STROKE_WIDTH = 2
ASCII_ESC = 27


class AppStatus(Enum):
    SUCCESS = 0
    GENERIC_FAILURE = -1


def get_object_bboxes(model_outputs):
    bbox_locations = []
    confidence_values = []

    for output in model_outputs:
        for prediction in output:
            car_probability = prediction[YOLO_PREDICTION_START + COCO_CLASSES_CAR_INDEX]
            if car_probability > THRESHOLD:
                w = int(prediction[YOLO_PREDICTION_W_POS] * YOLO_IMAGE_SIZE)
                h = int(prediction[YOLO_PREDICTION_H_POS] * YOLO_IMAGE_SIZE)
                x = int(prediction[YOLO_PREDICTION_X_POS] * YOLO_IMAGE_SIZE - w / 2)
                y = int(prediction[YOLO_PREDICTION_Y_POS] * YOLO_IMAGE_SIZE - h / 2)
                bbox_locations.append([x, y, w, h])
                confidence_values.append(car_probability)

    indices_to_keep = cv2.dnn.NMSBoxes(bbox_locations, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)
    filtered_bboxes = [bbox_locations[i] for i in indices_to_keep]
    filtered_confidence_values = [confidence_values[i] for i in indices_to_keep]

    return filtered_bboxes, filtered_confidence_values


def add_bboxes_to_frame(image, bboxes, confidence_values):
    height_ratio = image.shape[0] / YOLO_IMAGE_SIZE
    width_ratio = image.shape[1] / YOLO_IMAGE_SIZE
    for i, box in enumerate(bboxes):
        x = int(box[0] * width_ratio)
        y = int(box[1] * height_ratio)
        w = int(box[2] * width_ratio)
        h = int(box[3] * height_ratio)

        cv2.rectangle(image, (x, y), (x+w, y+h), CV2_BLUE, BOX_STROKE_WIDTH)
        class_and_conf_str = f"CAR {confidence_values[i] * 100:.1f}%"
        cv2.putText(image, class_and_conf_str, (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, CV2_BLUE)


def cleanup_resources(capture):
    capture.release()
    cv2.destroyAllWindows()


def main():
    nn = cv2.dnn.readNetFromDarknet(cfgFile="yolov3.cfg", darknetModel="yolov3.weights")
    nn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    nn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    capture = cv2.VideoCapture("cars_at_intersection.MOV")
    if not capture.isOpened():
        print("Error opening video capture, exiting...")
        cleanup_resources(capture)
        return AppStatus.GENERIC_FAILURE

    while True:
        grabbed, frame = capture.read()
        if not grabbed:
            break

        blob = cv2.dnn.blobFromImage(frame, YOLO_SCALE, (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE),
                                     swapRB=True, crop=False)
        nn.setInput(blob)
        layer_names = nn.getLayerNames()
        output_names = [layer_names[i - 1] for i in nn.getUnconnectedOutLayers()]
        outputs = nn.forward(output_names)
        car_bboxes, car_confidence_values = get_object_bboxes(outputs)
        add_bboxes_to_frame(frame, car_bboxes, car_confidence_values)

        cv2.imshow("Car Vision Video - San Diego Intersection", frame)
        if cv2.waitKey(1) & 0xFF == ASCII_ESC:
            break

    cleanup_resources(capture)
    return AppStatus.SUCCESS


if __name__ == "__main__":
    main()
