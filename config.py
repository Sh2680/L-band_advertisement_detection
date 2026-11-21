import os
import cv2

# ============================================
# DATABASE CONFIGURATION
# ============================================
DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Shilpa@123',
    'database': 'lband_ads',
    'pool_name': 'lband_pool',
    'pool_size': 5,
    'connection_timeout': 10,
    'autocommit': True
}

# ============================================
# CAMERA CONFIGURATION
# ============================================
CAMERA_CONFIG = {
    'source': 0,  
    'fps': 30,
    'buffer_size': 1,  
    'width': 640,  
    'height': 480,
    'backend': 'CAP_DSHOW',  
    'read_timeout_ms': 1000
}

# ============================================
# YOLO DETECTOR CONFIGURATION
# ============================================
YOLO_CONFIG = {
    'model_path': 'lband_best_model.pt',
    'confidence_threshold': 0.18,
    'iou_threshold': 0.5,
    'min_box_width': 120,
    'min_box_height': 120,
    'single_detection_mode': True,
    'device': 'cpu',  
    'img_size': 640,
    'half_precision': False,  
    'verbose': False
}

# ============================================
# DEEPSORT TRACKER CONFIGURATION
# ============================================
TRACKER_CONFIG = {
    'max_age': 30,  
    'n_init': 3,  
    'max_iou_distance': 0.7, 
    'embedder': 'mobilenet', 
    'half': False, 
    'embedder_gpu': False,  
    'max_cosine_distance': 0.3
}

# ============================================
# L-SHAPE BOUNDING BOX CONFIGURATION
# ============================================
LSHAPE_CONFIG = {
    'vertical_ratio': 133 / (133 + 529),  
    'horizontal_ratio': 104 / (104 + 522),  
    
    'color': (0, 255, 0), 
    'thickness': 3,
    'label_font': cv2.FONT_HERSHEY_SIMPLEX,
    'label_scale': 0.7,
    'label_thickness': 2,
    'label_offset_y': -10
}

# ============================================
# THREADING CONFIGURATION
# ============================================
THREAD_CONFIG = {
    'executor_max_workers': 2, 
    
    'frame_queue_size': 3,
    'detection_queue_size': 3,
    'result_queue_size': 3,
    
    'queue_get_timeout': 0.5,
    'queue_put_timeout': 0.1,
    'worker_shutdown_timeout': 5
}

# ============================================
# FASTAPI CONFIGURATION
# ============================================
API_CONFIG = {
    'host': '127.0.0.1',
    'port': 8000,
    'log_level': 'info',
    'reload': False
}

# ============================================
# DISPLAY CONFIGURATION
# ============================================
DISPLAY_CONFIG = {
    'window_name': 'L-Band Ads Detection System',
    'window_type': 'cv2.WINDOW_NORMAL',  
    'show_fps': True,
    'fps_position': (10, 30),
    'fps_color': (0, 255, 255),  
    'fps_scale': 0.7
}

# ============================================
# LOGGING CONFIGURATION
# ============================================
LOGGING_CONFIG = {
    'level': 'INFO',  
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S',
    'log_to_file': True,
    'log_file': 'lband_detection.log',
    'max_bytes': 10 * 1024 * 1024,  
    'backup_count': 5
}

# ============================================
# SYSTEM PATHS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, YOLO_CONFIG['model_path'])
LOG_PATH = os.path.join(BASE_DIR, LOGGING_CONFIG['log_file'])

# ============================================
# VALIDATION
# ============================================
def validate_config():
    errors = []

    if not os.path.exists(MODEL_PATH):
        errors.append(f"Model file not found: {MODEL_PATH}")

    if not isinstance(CAMERA_CONFIG['source'], (int, str)):
        errors.append("Camera source must be int (device index) or str (video path)")
    
    if not 0 <= YOLO_CONFIG['confidence_threshold'] <= 1:
        errors.append("Confidence threshold must be between 0 and 1")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    return True

if __name__ == '__main__':
    try:
        validate_config()
        print("✅ Configuration validated successfully!")
        print(f"📁 Base Directory: {BASE_DIR}")
        print(f"🎯 Model Path: {MODEL_PATH}")
        print(f"📝 Log Path: {LOG_PATH}")
    except ValueError as e:
        print(f"❌ Configuration validation failed:\n{e}")
