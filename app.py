import cv2
import time
import logging
from datetime import datetime
from queue import Queue, Empty, Full
import threading
import signal
import sys
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from config import (
    API_CONFIG, THREAD_CONFIG, DISPLAY_CONFIG, 
    LOGGING_CONFIG, validate_config
)
from camera import get_camera_stream
from detector import get_detector
from tracker import get_tracker
from database import get_database_manager

# ============================================
# LOGGING SETUP
# ============================================
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(LOGGING_CONFIG['format'])
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if LOGGING_CONFIG['log_to_file']:
        file_handler = RotatingFileHandler(
            LOGGING_CONFIG['log_file'],
            maxBytes=LOGGING_CONFIG['max_bytes'],
            backupCount=LOGGING_CONFIG['backup_count']
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(LOGGING_CONFIG['format'])
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

setup_logging()
logger = logging.getLogger(__name__)


# ============================================
# LIFESPAN EVENT HANDLER
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not start_system():
        logger.error("[ERROR] System startup failed")
        sys.exit(1)
    
    yield

    stop_system()


# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(title="L-Band Ads Detection System", lifespan=lifespan)


# ============================================
# GLOBAL STATE
# ============================================
class SystemState:
    def __init__(self):
        self.running = False
        self.camera = None
        self.detector = None
        self.tracker = None
        self.db_manager = None

        self.frame_queue = Queue(maxsize=THREAD_CONFIG['frame_queue_size'])
        self.detection_queue = Queue(maxsize=THREAD_CONFIG['detection_queue_size'])
        self.result_queue = Queue(maxsize=THREAD_CONFIG['result_queue_size'])

        self.active_ads = {}
        self.active_ads_lock = threading.Lock()
        
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'start_time': None
        }
        self.stats_lock = threading.Lock()

state = SystemState()


# ============================================
# WORKER THREADS
# ============================================

def camera_capture_worker():
    logger.info("[CAMERA] Camera capture worker started")
    
    while state.running:
        try:
            frame = state.camera.get_frame()
            
            if frame is None:
                time.sleep(0.01)
                continue

            try:
                state.frame_queue.put(frame, timeout=THREAD_CONFIG['queue_put_timeout'])
            except Full:

                pass
            
            with state.stats_lock:
                state.stats['total_frames'] += 1
            
        except Exception as e:
            logger.error(f"[ERROR] Camera worker error: {e}", exc_info=True)
            time.sleep(0.1)
    
    logger.info("[CAMERA] Camera capture worker stopped")


def detection_worker():
    logger.info("[*] Detection worker started")
    
    pending_future = None
    
    while state.running:
        try:
            if pending_future is not None:
                if pending_future.done():
                    try:
                        detections = pending_future.result()
 
                        try:
                            state.detection_queue.put(
                                (current_frame, detections),
                                timeout=THREAD_CONFIG['queue_put_timeout']
                            )
                            
                            with state.stats_lock:
                                state.stats['total_detections'] += len(detections)
                            
                        except Full:
                            pass  
                        
                        pending_future = None
                        
                    except Exception as e:
                        logger.error(f"[ERROR] Detection result error: {e}")
                        pending_future = None
            
            # Get new frame if no pending detection
            if pending_future is None:
                try:
                    current_frame = state.frame_queue.get(
                        timeout=THREAD_CONFIG['queue_get_timeout']
                    )
                    
                    pending_future = state.detector.detect_async(current_frame)
                    
                except Empty:
                    continue
            else:
                time.sleep(0.001)
            
        except Exception as e:
            logger.error(f"[ERROR] Detection worker error: {e}", exc_info=True)
            time.sleep(0.1)
            pending_future = None
    
    logger.info("[*] Detection worker stopped")


def tracking_worker():
    logger.info("[TARGET] Tracking worker started")
    
    pending_future = None
    
    while state.running:
        try:
            if pending_future is not None:
                if pending_future.done():
                    try:
                        annotated_frame, track_ids, track_info = pending_future.result()
                        
                        now = datetime.now()
                        current_ids = set(track_ids)
                        
                        with state.active_ads_lock:
                            for tid in current_ids:
                                if tid not in state.active_ads:
                                    state.active_ads[tid] = {
                                        'start': now,
                                        'count': 1
                                    }
                                    logger.info(f"[*] New L-Band detected: ID {tid}")
                                else:
                                    state.active_ads[tid]['count'] += 1
                            
                            # Remove and log disappeared tracks
                            for tid in list(state.active_ads.keys()):
                                if tid not in current_ids:
                                    info = state.active_ads.pop(tid)
                                    duration = (now - info['start']).total_seconds()
                                    
                                    if state.db_manager:
                                        try:
                                            state.db_manager.insert_ad_event(
                                                track_id=int(tid),
                                                start_time=info['start'],
                                                end_time=now,
                                                duration_seconds=duration
                                            )
                                        except Exception as e:
                                            logger.error(f"[ERROR] Failed to log ad {tid}: {e}")
                                    
                                    state.tracker.mark_track_completed(
                                        tid, duration, info['count']
                                    )
                        
                        # Put to result queue
                        try:
                            state.result_queue.put(
                                annotated_frame,
                                timeout=THREAD_CONFIG['queue_put_timeout']
                            )
                        except Full:
                            pass
                        
                        with state.stats_lock:
                            state.stats['total_tracks'] = state.tracker.get_total_track_count()
                        
                        pending_future = None
                        
                    except Exception as e:
                        logger.error(f"[ERROR] Tracking result error: {e}")
                        pending_future = None
            
            # Get new detection if no pending tracking
            if pending_future is None:
                try:
                    frame, detections = state.detection_queue.get(
                        timeout=THREAD_CONFIG['queue_get_timeout']
                    )
                    
                    # Submit tracking asynchronously
                    pending_future = state.tracker.track_and_draw_async(frame, detections)
                    
                except Empty:
                    continue
            else:
                time.sleep(0.001)
            
        except Exception as e:
            logger.error(f"[ERROR] Tracking worker error: {e}", exc_info=True)
            time.sleep(0.1)
            pending_future = None
    
    logger.info("[TARGET] Tracking worker stopped")


# ============================================
# SYSTEM CONTROL
# ============================================

def start_system():
    """Start all system components"""
    logger.info("[START] Starting L-Band Detection System...")
    
    try:
        # Validate configuration
        validate_config()
        
        # Initialize components
        state.camera = get_camera_stream()
        state.detector = get_detector()
        state.tracker = get_tracker()
        state.db_manager = get_database_manager()
        
        # Start camera
        if not state.camera.start():
            raise RuntimeError("Failed to start camera")
        
        # Set running flag
        state.running = True
        state.stats['start_time'] = datetime.now()
        
        # Start worker threads
        threading.Thread(target=camera_capture_worker, daemon=True, name="CameraWorker").start()
        threading.Thread(target=detection_worker, daemon=True, name="DetectionWorker").start()
        threading.Thread(target=tracking_worker, daemon=True, name="TrackingWorker").start()
        
        logger.info("[OK] System started successfully")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to start system: {e}", exc_info=True)
        return False


def stop_system():
    logger.info("[STOP] Stopping L-Band Detection System...")
    
    state.running = False
    
    # Stop camera
    if state.camera:
        state.camera.stop()
    
    # Shutdown detector
    if state.detector:
        state.detector.shutdown()
    
    # Shutdown tracker
    if state.tracker:
        state.tracker.shutdown()
    
    # Close database pool
    if state.db_manager:
        state.db_manager.close_pool()
    
    logger.info("[OK] System stopped")


# ============================================
# FASTAPI ENDPOINTS
# ============================================

@app.get("/")
def health_check():
    uptime = None
    start_time_str = None
    
    if state.stats['start_time']:
        uptime = (datetime.now() - state.stats['start_time']).total_seconds()
        start_time_str = state.stats['start_time'].isoformat()
    
    # Create serializable stats
    serializable_stats = {
        'total_frames': state.stats['total_frames'],
        'total_detections': state.stats['total_detections'],
        'total_tracks': state.stats['total_tracks'],
        'start_time': start_time_str
    }
    
    return JSONResponse({
        "status": "running" if state.running else "stopped",
        "uptime_seconds": uptime,
        "camera_fps": state.camera.get_fps() if state.camera else 0,
        "queues": {
            "frame_queue": state.frame_queue.qsize(),
            "detection_queue": state.detection_queue.qsize(),
            "result_queue": state.result_queue.qsize()
        },
        "active_ads": len(state.active_ads),
        "daily_occurrences": state.db_manager.get_daily_occurrences() if state.db_manager else 0,
        "stats": serializable_stats
    })


@app.get("/statistics")
def get_statistics():
    detector_stats = state.detector.get_statistics() if state.detector else {}
    tracker_stats = state.tracker.get_statistics() if state.tracker else {}
    db_stats = state.db_manager.get_statistics() if state.db_manager else {}
    
    # Create serializable stats
    serializable_stats = {
        'total_frames': state.stats['total_frames'],
        'total_detections': state.stats['total_detections'],
        'total_tracks': state.stats['total_tracks'],
        'start_time': state.stats['start_time'].isoformat() if state.stats['start_time'] else None
    }
    
    return JSONResponse({
        "system": serializable_stats,
        "detector": detector_stats,
        "tracker": tracker_stats,
        "database": db_stats,
        "active_ads": list(state.active_ads.keys())
    })


@app.get("/recent-events")
def get_recent_events(limit: int = 10):
    if state.db_manager:
        events = state.db_manager.get_recent_events(limit)
        
        # Convert datetime objects to ISO format strings
        serializable_events = []
        for event in events:
            serializable_event = {}
            for key, value in event.items():
                if isinstance(value, datetime):
                    serializable_event[key] = value.isoformat()
                else:
                    serializable_event[key] = value
            serializable_events.append(serializable_event)
        
        return JSONResponse({"events": serializable_events})
    return JSONResponse({"error": "Database not available"}, status_code=503)


# ============================================
# SIGNAL HANDLERS
# ============================================

def signal_handler(sig, frame):
    logger.info("[STOP] Received interrupt signal, shutting down...")
    state.running = False
    stop_system()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================
# MAIN EXECUTION
# ============================================

def run_display_window():
    logger.info("[DISPLAY] Starting display window...")
    
    cv2.namedWindow(DISPLAY_CONFIG['window_name'], cv2.WINDOW_NORMAL)
    
    last_frame = None
    no_frame_count = 0
    fps_update_time = time.time()
    display_fps = 0
    frame_count_for_fps = 0
    
    try:
        while state.running:
            try:
                # Get result frame
                try:
                    frame = state.result_queue.get(timeout=0.05)
                    last_frame = frame
                    no_frame_count = 0
                    frame_count_for_fps += 1
                except Empty:
                    frame = last_frame
                    no_frame_count += 1
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Calculate display FPS
                current_time = time.time()
                if current_time - fps_update_time >= 1.0:
                    display_fps = frame_count_for_fps
                    frame_count_for_fps = 0
                    fps_update_time = current_time
                
                # Add system info overlay
                if DISPLAY_CONFIG['show_fps']:
                    cam_fps = state.camera.get_fps() if state.camera else 0
                    today_occurrences = state.db_manager.get_daily_occurrences() if state.db_manager else 0
                    info_text = f"FPS: {display_fps} | Camera: {cam_fps} | Active: {len(state.active_ads)} | Today: {today_occurrences}"
                    
                    cv2.putText(
                        frame, info_text,
                        DISPLAY_CONFIG['fps_position'],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        DISPLAY_CONFIG['fps_scale'],
                        DISPLAY_CONFIG['fps_color'],
                        2
                    )
                
                # Show frame
                cv2.imshow(DISPLAY_CONFIG['window_name'], frame)
                
                # Check for quit via 'q' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("[*] User pressed 'q', exiting...")
                    state.running = False
                    break
                
                # Check if window was closed using the Windows 'X' close button
                if cv2.getWindowProperty(DISPLAY_CONFIG['window_name'], cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("[*] Display window closed by user, exiting...")
                    state.running = False
                    break
                
                # Warn if no frames
                if no_frame_count > 100:
                    logger.warning("[WARNING] No frames received for 5 seconds")
                    no_frame_count = 0
                
            except Exception as e:
                logger.error(f"[ERROR] Display loop error: {e}", exc_info=True)
                time.sleep(0.1)
    
    finally:
        cv2.destroyAllWindows()
        logger.info("[DISPLAY] Display window closed")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("L-BAND ADS DETECTION SYSTEM")
    logger.info("=" * 60)
    
    # Start FastAPI in background thread
    def run_api():
        uvicorn.run(
            app,
            host=API_CONFIG['host'],
            port=API_CONFIG['port'],
            log_level=API_CONFIG['log_level'].lower()
        )
    
    api_thread = threading.Thread(target=run_api, daemon=True, name="FastAPI")
    api_thread.start()
    
    # Wait dynamically for system initialization instead of fixed sleep
    logger.info("[WAIT] Waiting for system initialization (camera, model, database)...")
    timeout_seconds = 45
    start_wait = time.time()
    while not state.running and api_thread.is_alive():
        if time.time() - start_wait > timeout_seconds:
            logger.error("[ERROR] System initialization timed out after 45s")
            break
        time.sleep(0.2)
    
    if state.running:
        run_display_window()
    else:
        logger.error("[ERROR] Application startup failed. See logs above for details.")
    
    # Cleanup
    stop_system()
    logger.info("[OK] Application terminated cleanly")