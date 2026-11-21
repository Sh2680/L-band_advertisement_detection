import cv2
import threading
import time
import logging
import numpy as np
from queue import Queue, Full
from config import CAMERA_CONFIG

logger = logging.getLogger(__name__)


class CameraStream:
    
    def __init__(self, source=None):
        self.source = source if source is not None else CAMERA_CONFIG['source']
        self.cap = None
        
        self.frame = None
        self.frame_lock = threading.Lock()
        
        self.running = False
        self.thread = None
        
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Error handling
        self.consecutive_failures = 0
        self.max_consecutive_failures = 30
        
        logger.info(f"[CAMERA] Camera stream initialized with source: {self.source}")
    
    def initialize_camera(self):
        try:
            backend_map = {
                'CAP_DSHOW': cv2.CAP_DSHOW,
                'CAP_V4L2': cv2.CAP_V4L2,
                'CAP_ANY': cv2.CAP_ANY
            }
            backend = backend_map.get(CAMERA_CONFIG.get('backend', 'CAP_DSHOW'), cv2.CAP_ANY)
            
            self.cap = cv2.VideoCapture(self.source, backend)
            
            if not self.cap.isOpened():
                logger.error(f"[ERROR] Failed to open camera source: {self.source}")
                return False

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_CONFIG['buffer_size'])
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
            
            if 'width' in CAMERA_CONFIG and 'height' in CAMERA_CONFIG:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['width'])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['height'])
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(
                f"[OK] Camera opened: {actual_width}x{actual_height} @ {actual_fps}fps"
            )
            
            self.consecutive_failures = 0
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Camera initialization error: {e}")
            return False
    
    def capture_loop(self):
        logger.info("[CAMERA] Camera capture thread started")
        
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logger.warning("[WARNING] Camera not available, attempting to reinitialize...")
                    if not self.initialize_camera():
                        time.sleep(1)
                        continue
                
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.consecutive_failures += 1
                    logger.warning(f"[WARNING] Failed to read frame ({self.consecutive_failures})")
                    
                    if self.consecutive_failures > self.max_consecutive_failures:
                        logger.error("[ERROR] Too many consecutive failures, restarting camera...")
                        self.release_camera()
                        time.sleep(2)
                        self.consecutive_failures = 0
                    
                    time.sleep(0.01)
                    continue
                
                # Success - update shared memory
                with self.frame_lock:
                    self.frame = frame.copy()
                    self.frame_count += 1
                
                self.consecutive_failures = 0
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # Small sleep to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"[ERROR] Error in capture loop: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("[CAMERA] Camera capture thread stopped")
    
    def start(self):
        if self.running:
            logger.warning("[WARNING] Camera already running")
            return False
        
        if not self.initialize_camera():
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()
        
        logger.info("[OK] Camera stream started")
        return True
    
    def stop(self):
        if not self.running:
            return
        
        logger.info("[STOP] Stopping camera stream...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=3)
        
        self.release_camera()
        logger.info("[OK] Camera stream stopped")
    
    def release_camera(self):
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
                logger.info("[OK] Camera released")
            except Exception as e:
                logger.error(f"[ERROR] Error releasing camera: {e}")
    
    def get_frame(self):
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps
    
    def is_running(self):
        """Check if camera is running"""
        return self.running and self.thread and self.thread.is_alive()


# Singleton instance
_camera_stream = None

def get_camera_stream():
    """Get singleton camera stream instance"""
    global _camera_stream
    if _camera_stream is None:
        _camera_stream = CameraStream()
    return _camera_stream


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    print("[*] Testing Camera Module...")
    
    camera = CameraStream()
    
    if not camera.start():
        print("[ERROR] Failed to start camera")
        exit(1)
    
    print("[OK] Camera started. Press 'q' to quit...")
    
    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame = camera.get_frame()
            
            if frame is not None:
                # Display FPS
                fps_text = f"FPS: {camera.get_fps()}"
                cv2.putText(frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("Camera Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("[OK] Camera test completed")