import cv2
import torch
import logging
import time
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple
from config import YOLO_CONFIG, THREAD_CONFIG, MODEL_PATH

logger = logging.getLogger(__name__)


class LBandDetector:
    
    def __init__(self):
        self.model = None
        self.device = None
        self.executor = None
        self.is_initialized = False
        
        # Statistics
        self.total_detections = 0
        self.total_frames_processed = 0
        self.avg_inference_time = 0
        
        self.initialize()
    
    def initialize(self):
        try:
            logger.info(f"[LOADING] Loading YOLO11 model from: {MODEL_PATH}")
            self.model = YOLO(MODEL_PATH)

            if YOLO_CONFIG['device'].startswith('cuda') and torch.cuda.is_available():
                self.device = YOLO_CONFIG['device']
                logger.info(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                logger.warning("[WARNING] CUDA not available, using CPU")
            

            self.executor = ThreadPoolExecutor(
                max_workers=1,  # Single worker for detector
                thread_name_prefix="YOLO_Detector"
            )
            
            self.is_initialized = True
            logger.info("[OK] L-Band detector initialized successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize detector: {e}", exc_info=True)
            self.is_initialized = False
            raise
    
    def detect_sync(self, frame) -> List[List[float]]:

        if frame is None:
            return []
        
        if not self.is_initialized:
            logger.error("[ERROR] Detector not initialized")
            return []
        
        try:
            start_time = time.time()
            
            # Run YOLO prediction
            results = self.model.predict(
                source=frame,
                conf=YOLO_CONFIG['confidence_threshold'],
                iou=YOLO_CONFIG['iou_threshold'],
                device=self.device,
                verbose=YOLO_CONFIG['verbose'],
                half=YOLO_CONFIG['half_precision'],
                imgsz=YOLO_CONFIG['img_size']
            )
            
            # Extract detections
            candidates = []
            min_w = YOLO_CONFIG.get('min_box_width', 120)
            min_h = YOLO_CONFIG.get('min_box_height', 120)
            
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                
                for box in result.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # DeepSORT format: ([left, top, width, height], confidence, class_id)
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    
                    # Size filter: ignore small noisy patches/logos inside the banner
                    if w >= min_w and h >= min_h:
                        candidates.append(([float(x1), float(y1), w, h], confidence, class_id, w * h))
            
            detections = []
            if candidates:
                if YOLO_CONFIG.get('single_detection_mode', True):
                    # Pick the single largest L-band ad on the screen
                    best_candidate = max(candidates, key=lambda c: c[3])
                    detections.append(best_candidate[:3])
                else:
                    detections = [c[:3] for c in candidates]
            
            # Update statistics
            inference_time = time.time() - start_time
            self.total_frames_processed += 1
            self.total_detections += len(detections)
            self.avg_inference_time = (
                (self.avg_inference_time * (self.total_frames_processed - 1) + inference_time)
                / self.total_frames_processed
            )
            
            if len(detections) > 0:
                logger.debug(
                    f"[TARGET] Detected {len(detections)} L-bands in {inference_time*1000:.1f}ms"
                )
            
            return detections
            
        except Exception as e:
            logger.error(f"[ERROR] Detection error: {e}", exc_info=True)
            return []
    
    def detect_async(self, frame) -> Future:
        if not self.is_initialized:
            logger.error("[ERROR] Detector not initialized")
            # Return a future with empty results
            future = Future()
            future.set_result([])
            return future
        
        return self.executor.submit(self.detect_sync, frame)
    
    def get_statistics(self) -> dict:
        return {
            'total_frames_processed': self.total_frames_processed,
            'total_detections': self.total_detections,
            'avg_inference_time_ms': self.avg_inference_time * 1000,
            'fps': 1.0 / self.avg_inference_time if self.avg_inference_time > 0 else 0
        }
    
    def shutdown(self):
        logger.info("[STOP] Shutting down detector...")
        
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("[OK] Detector executor shutdown complete")
        
        self.is_initialized = False


# Singleton instance
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = LBandDetector()
    return _detector


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("[*] Testing Detector Module...")
    
    try:
        detector = LBandDetector()
        
        import os
        if os.path.exists('test_image.jpg'):
            test_frame = cv2.imread('test_image.jpg')
        else:
            test_frame = (255 * torch.rand(640, 640, 3)).numpy().astype('uint8')
            print("[WARNING] Using random test frame (no test_image.jpg found)")
        
        # Synchronous detection
        print("\n[STATS] Testing synchronous detection...")
        detections = detector.detect_sync(test_frame)
        print(f"[OK] Found {len(detections)} detections")
        
        # Asynchronous detection
        print("\n[STATS] Testing asynchronous detection...")
        future = detector.detect_async(test_frame)
        detections = future.result(timeout=5)
        print(f"[OK] Found {len(detections)} detections (async)")
        
        # Statistics
        stats = detector.get_statistics()
        print(f"\n[*] Statistics:")
        print(f"   Frames processed: {stats['total_frames_processed']}")
        print(f"   Total detections: {stats['total_detections']}")
        print(f"   Avg inference time: {stats['avg_inference_time_ms']:.2f}ms")
        print(f"   Detection FPS: {stats['fps']:.2f}")
        
        detector.shutdown()
        print("\n[OK] Detector test completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Detector test failed: {e}")
        import traceback
        traceback.print_exc()