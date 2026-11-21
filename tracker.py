import cv2
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Tuple
from config import TRACKER_CONFIG, LSHAPE_CONFIG, THREAD_CONFIG

logger = logging.getLogger(__name__)


class LBandTracker:
    
    def __init__(self):
        self.tracker = None
        self.executor = None
        self.is_initialized = False
        
        # Statistics
        self.total_tracks = 0
        self.active_tracks = set()
        self.completed_tracks = {}
        
        self.initialize()
    
    def initialize(self):
        try:
            logger.info("[LOADING] Initializing DeepSORT tracker...")
            
            # Initialize DeepSORT
            self.tracker = DeepSort(
                max_age=TRACKER_CONFIG['max_age'],
                n_init=TRACKER_CONFIG['n_init'],
                max_iou_distance=TRACKER_CONFIG['max_iou_distance'],
                embedder=TRACKER_CONFIG['embedder'],
                half=TRACKER_CONFIG['half'],
                embedder_gpu=TRACKER_CONFIG['embedder_gpu'],
                max_cosine_distance=TRACKER_CONFIG['max_cosine_distance']
            )
            
            self.executor = ThreadPoolExecutor(
                max_workers=1, 
                thread_name_prefix="DeepSORT_Tracker"
            )
            
            self.is_initialized = True
            logger.info("[OK] L-Band tracker initialized successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize tracker: {e}", exc_info=True)
            self.is_initialized = False
            raise
    
    def draw_lshape_box(self, frame, x1, y1, x2, y2, track_id, confidence=None):
        """
        Draw L-shaped bounding box strictly limited to detection boundary on frame
        
        Args:
            frame: Image frame
            x1, y1, x2, y2: Bounding box coordinates
            track_id: Tracking ID
            confidence: Detection confidence (optional)
        """
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, min(int(x1), w_frame - 1))
        y1 = max(0, min(int(y1), h_frame - 1))
        x2 = max(0, min(int(x2), w_frame - 1))
        y2 = max(0, min(int(y2), h_frame - 1))

        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            return

        vertical_width = int(box_width * LSHAPE_CONFIG['vertical_ratio'])
        horizontal_height = int(box_height * LSHAPE_CONFIG['horizontal_ratio'])
        vertical_width = max(2, min(vertical_width, box_width))
        horizontal_height = max(2, min(horizontal_height, box_height))
        
        color = LSHAPE_CONFIG['color']
        thickness = LSHAPE_CONFIG['thickness']
        
        # Draw vertical bar (left side)
        cv2.rectangle(
            frame,
            (x1, y1),
            (x1 + vertical_width, y2),
            color,
            thickness
        )
        
        # Draw horizontal bar (bottom)
        cv2.rectangle(
            frame,
            (x1, y2 - horizontal_height),
            (x2, y2),
            color,
            thickness
        )
        
        # Prepare label text
        label = f"L-Band ID:{track_id}"
        if confidence is not None:
            label += f" ({confidence:.2f})"
        
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = LSHAPE_CONFIG['label_scale']
        font_thickness = LSHAPE_CONFIG['label_thickness']
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        label_y = y1 + LSHAPE_CONFIG['label_offset_y']
        if label_y - text_height - 5 < 0:
            label_y = y1 + text_height + 15
        
        # Clamp label background within frame bounds
        x1_label_bg = max(0, min(x1, w_frame - text_width - 12))
        x2_label_bg = min(w_frame - 1, x1_label_bg + text_width + 10)
        y1_label_bg = max(0, label_y - text_height - 5)
        y2_label_bg = min(h_frame - 1, label_y + 5)
        
        # Background rectangle for text
        cv2.rectangle(
            frame,
            (x1_label_bg, y1_label_bg),
            (x2_label_bg, y2_label_bg),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1_label_bg + 5, label_y),
            font,
            font_scale,
            (0, 0, 0),  # Black text
            font_thickness
        )
    
    def track_and_draw_sync(self, frame, detections) -> Tuple[np.ndarray, List[int], List[dict]]:
        """
        Synchronous tracking and drawing
        
        Args:
            frame: Input frame
            detections: List of detections [[x1, y1, x2, y2, confidence], ...]
        
        Returns:
            Tuple of (annotated_frame, track_ids, track_info)
        """
        if frame is None:
            return frame, [], []
        
        if not self.is_initialized:
            logger.error("[ERROR] Tracker not initialized")
            return frame, [], []
        
        try:
            # Update tracks with DeepSORT
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            track_ids = []
            track_info = []
            
            for track in tracks:
                # Only process confirmed tracks
                if not track.is_confirmed():
                    continue
                
                try:
                    track_id = int(track.track_id)
                except (ValueError, TypeError):
                    track_id = track.track_id
                track_ids.append(track_id)
                
                # Get bounding box
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                
                # Get confidence if available
                confidence = None
                if hasattr(track, 'get_det_conf') and callable(track.get_det_conf):
                    confidence = track.get_det_conf()
                elif hasattr(track, 'det_conf') and track.det_conf is not None:
                    confidence = track.det_conf
                
                # Draw L-shaped bounding box
                self.draw_lshape_box(frame, x1, y1, x2, y2, track_id, confidence)
                
                # Store track info
                track_info.append({
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'state': 'confirmed'
                })
                
                # Update active tracks
                self.active_tracks.add(track_id)
            
            return frame, track_ids, track_info
            
        except Exception as e:
            logger.error(f"[ERROR] Tracking error: {e}", exc_info=True)
            return frame, [], []
    
    def track_and_draw_async(self, frame, detections) -> Future:
        """
        Asynchronous tracking using ThreadPoolExecutor
        
        Args:
            frame: Input frame
            detections: List of detections
        
        Returns:
            Future object containing (annotated_frame, track_ids, track_info)
        """
        if not self.is_initialized:
            logger.error("[ERROR] Tracker not initialized")
            future = Future()
            future.set_result((frame, [], []))
            return future
        
        return self.executor.submit(self.track_and_draw_sync, frame, detections)
    
    def get_active_track_count(self) -> int:
        return len(self.active_tracks)
    
    def get_total_track_count(self) -> int:
        return len(self.active_tracks) + len(self.completed_tracks)
    
    def mark_track_completed(self, track_id, duration, frame_count):
        if track_id in self.active_tracks:
            self.active_tracks.remove(track_id)
            self.completed_tracks[track_id] = {
                'duration': duration,
                'frame_count': frame_count
            }
            logger.info(
                f"[OK] Track {track_id} completed: {duration:.2f}s, {frame_count} frames"
            )
    
    def get_statistics(self) -> dict:
        return {
            'active_tracks': len(self.active_tracks),
            'completed_tracks': len(self.completed_tracks),
            'total_tracks': self.get_total_track_count()
        }
    
    def shutdown(self):
        logger.info("[STOP] Shutting down tracker...")
        
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("[OK] Tracker executor shutdown complete")
        
        self.is_initialized = False


# Singleton instance
_tracker = None

def get_tracker():
    """Get singleton tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = LBandTracker()
    return _tracker


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("[*] Testing Tracker Module...")
    
    try:
        tracker = LBandTracker()

        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        test_detections = [
            ([100, 100, 200, 300], 0.95, 0),  
            ([500, 200, 200, 300], 0.88, 0),  
        ]
        
        # Synchronous tracking
        print("\n[STATS] Testing synchronous tracking...")
        annotated_frame, track_ids, track_info = tracker.track_and_draw_sync(
            test_frame.copy(), test_detections
        )
        print(f"[OK] Tracked {len(track_ids)} objects: {track_ids}")
        print(f"   Track info: {track_info}")
        
        # Asynchronous tracking
        print("\n[STATS] Testing asynchronous tracking...")
        future = tracker.track_and_draw_async(test_frame.copy(), test_detections)
        annotated_frame, track_ids, track_info = future.result(timeout=5)
        print(f"[OK] Tracked {len(track_ids)} objects (async): {track_ids}")
        
        # Statistics
        stats = tracker.get_statistics()
        print(f"\n[*] Statistics:")
        print(f"   Active tracks: {stats['active_tracks']}")
        print(f"   Completed tracks: {stats['completed_tracks']}")
        print(f"   Total tracks: {stats['total_tracks']}")
        
        # Test L-shape drawing
        print("\n[*] Testing L-shape drawing...")
        tracker.draw_lshape_box(test_frame, 400, 300, 600, 600, track_id=99, confidence=0.92)
        cv2.imwrite('test_lshape.png', test_frame)
        print("[OK] L-shape test image saved as 'test_lshape.png'")
        
        tracker.shutdown()
        print("\n[OK] Tracker test completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Tracker test failed: {e}")
        import traceback
        traceback.print_exc()