from ultralytics import YOLO

def load_model(weights_path):
    return YOLO(weights_path)

def run_detection(model, image_path):
    return model(image_path)