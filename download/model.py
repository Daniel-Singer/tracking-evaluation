import os
from ultralytics import YOLO

def define_model(version=None):
    
    model = YOLO(version)
    
    return model