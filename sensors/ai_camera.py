#!/usr/bin/env python3
"""
AI-Powered Camera Module with Animal/Human Detection
Optimized for Raspberry Pi 4 (1GB RAM) with YOLOv5 Nano
BloomBotanics - October 2025
"""

import os
import time
import cv2
import numpy as np
from picamera2 import Picamera2
import torch
from utils.logger import get_logger
from config import *


class AICamera:
    """AI Camera for threat detection using YOLOv5 Nano"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.camera = None
        self.model = None
        self.last_detection_time = 0
        self.detection_count = 0
        
        # Initialize components
        self.initialize_camera()
        if AI_DETECTION_ENABLED:
            self.initialize_ai_model()
    
    def initialize_camera(self):
        """Initialize Pi Camera with optimized settings for AI"""
        try:
            self.camera = Picamera2()
            
            # Optimized configuration for low memory
            config = self.camera.create_still_configuration(
                main={"size": CAMERA_RESOLUTION},  # Low resolution from config
                lores={"size": (320, 240)},        # Even lower for preview
                display="lores"
            )
            
            self.camera.configure(config)
            self.camera.start()
            
            # Camera warm-up
            time.sleep(2)
            self.logger.info(f"Camera initialized: {CAMERA_RESOLUTION}")
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            self.camera = None
    
    def initialize_ai_model(self):
        """Initialize YOLOv5 Nano model optimized for Raspberry Pi"""
        try:
            # Use YOLOv5 Nano for edge devices
            model_name = "yolov5n"  # Smallest, fastest YOLO model
            
            self.logger.info(f"Loading {model_name} model...")
            
            # Load YOLOv5 from torch hub (will download on first run)
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            
            # Optimize for CPU (Pi 4 doesn't have GPU)
            self.model.cpu()
            self.model.conf = DETECTION_CONFIDENCE  # Set confidence threshold
            self.model.iou = 0.45  # IoU threshold for NMS
            self.model.max_det = 10  # Maximum detections per image
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Warm-up model with dummy inference
            dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = self.model(dummy_img)
            
            self.logger.info(f"AI model {model_name} loaded successfully")
            
        except Exception as e:
            self.logger.error(f"AI model initialization failed: {e}")
            self.logger.error("Install with: pip install torch torchvision ultralytics")
            self.model = None
    
    def capture_photo(self, filename=None):
        """Capture regular photo without AI processing"""
        if not self.camera:
            self.logger.warning("Camera not initialized")
            return None
        
        try:
            if not filename:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(IMAGE_DIR, f'crop_{timestamp}.jpg')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Capture photo
            self.camera.capture_file(filename)
            self.logger.info(f"Photo captured: {os.path.basename(filename)}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Photo capture error: {e}")
            return None
    
    def detect_threats(self):
        """AI detection of animals, humans, and intruders"""
        if not self.camera or not self.model:
            return [], None
        
        # Rate limiting - don't run AI detection too frequently
        current_time = time.time()
        if current_time - self.last_detection_time < DETECTION_INTERVAL:
            return [], None
        
        try:
            # Capture frame for detection
            frame = self.camera.capture_array()
            
            # Convert from RGB to BGR for OpenCV compatibility
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize frame for better performance on Pi (320x320 or 416x416)
            detection_size = 320  # Smaller = faster but less accurate
            height, width = frame.shape[:2]
            
            if width > detection_size or height > detection_size:
                frame_resized = cv2.resize(frame, (detection_size, detection_size))
            else:
                frame_resized = frame
            
            # Run YOLOv5 detection
            self.logger.info("Running AI detection...")
            results = self.model(frame_resized)
            
            # Parse results
            detections = []
            predictions = results.pandas().xyxy[0]  # Get predictions as pandas DataFrame
            
            for idx, pred in predictions.iterrows():
                class_id = int(pred['class'])
                confidence = float(pred['confidence'])
                
                # Check if detected class is a threat
                all_threats = {**CRITICAL_THREATS, **MONITOR_THREATS}
                
                if class_id in all_threats:
                    # Scale bounding box coordinates back to original frame size
                    scale_x = width / detection_size
                    scale_y = height / detection_size
                    
                    x1 = int(pred['xmin'] * scale_x)
                    y1 = int(pred['ymin'] * scale_y)
                    x2 = int(pred['xmax'] * scale_x)
                    y2 = int(pred['ymax'] * scale_y)
                    
                    # Calculate detection area
                    bbox_area = (x2 - x1) * (y2 - y1)
                    frame_area = width * height
                    relative_size = bbox_area / frame_area
                    
                    detection = {
                        'class_id': class_id,
                        'type': all_threats[class_id]['name'],
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'relative_size': relative_size,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'threat_level': 'critical' if class_id in CRITICAL_THREATS else 'monitor',
                        'action': all_threats[class_id]['action']
                    }
                    detections.append(detection)
                    
                    self.logger.info(f"🚨 Detected: {detection['type']} "
                                   f"(confidence: {confidence:.2f}, "
                                   f"level: {detection['threat_level']})")
            
            self.last_detection_time = current_time
            self.detection_count += len(detections)
            
            return detections, frame
            
        except Exception as e:
            self.logger.error(f"AI detection error: {e}")
            return [], None
    
    def save_detection_image(self, frame, detections, filename=None):
        """Save image with detection boxes and labels"""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            threat_types = "_".join([d['type'] for d in detections[:3]])  # Max 3 in filename
            filename = os.path.join(DETECTION_DIR, f'threat_{threat_types}_{timestamp}.jpg')
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Draw detection boxes and labels
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                
                # Color based on threat level
                if detection['threat_level'] == 'critical':
                    color = (0, 0, 255)  # Red for critical
                    thickness = 3
                else:
                    color = (0, 255, 255)  # Yellow for monitoring
                    thickness = 2
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Prepare label
                label = f"{detection['type']} {detection['confidence']:.2f}"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 10, y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add header with system info
            header_bg = np.zeros((40, frame.shape[1], 3), dtype=np.uint8)
            header_bg[:] = (0, 0, 0)
            
            timestamp_text = f"BloomBotanics | {time.strftime('%Y-%m-%d %H:%M:%S')} | Detections: {len(detections)}"
            cv2.putText(header_bg, timestamp_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Combine header and frame
            frame_with_header = np.vstack([header_bg, frame])
            
            # Save annotated image
            cv2.imwrite(filename, frame_with_header)
            self.logger.info(f"Detection image saved: {os.path.basename(filename)}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Detection image save error: {e}")
            return None
    
    def get_detection_statistics(self):
        """Get detection statistics for reporting"""
        try:
            detection_files = [f for f in os.listdir(DETECTION_DIR) 
                             if f.startswith('threat_') and f.endswith('.jpg')]
            
            today_str = time.strftime('%Y%m%d')
            detections_today = [f for f in detection_files if today_str in f]
            
            return {
                'total_detections': self.detection_count,
                'total_images': len(detection_files),
                'detections_today': len(detections_today),
                'last_detection': max(detection_files) if detection_files else None,
                'last_detection_time': time.strftime('%Y-%m-%d %H:%M:%S', 
                                                    time.localtime(self.last_detection_time))
                                     if self.last_detection_time > 0 else "Never"
            }
        except Exception as e:
            self.logger.error(f"Statistics error: {e}")
            return {
                'total_detections': 0,
                'total_images': 0,
                'detections_today': 0,
                'last_detection': None,
                'last_detection_time': "Never"
            }
    
    def run_continuous_detection(self, duration=60, callback=None):
        """
        Run continuous threat detection for a specified duration
        Args:
            duration: Detection duration in seconds
            callback: Function to call when threats are detected
        """
        start_time = time.time()
        detection_log = []
        
        self.logger.info(f"Starting continuous detection for {duration} seconds...")
        
        while time.time() - start_time < duration:
            detections, frame = self.detect_threats()
            
            if detections:
                # Save detection image
                image_path = self.save_detection_image(frame, detections)
                
                # Log detection
                detection_log.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'detections': detections,
                    'image_path': image_path
                })
                
                # Call callback if provided
                if callback:
                    callback(detections, frame, image_path)
            
            # Wait before next detection
            time.sleep(DETECTION_INTERVAL)
        
        self.logger.info(f"Continuous detection complete. Total threats: {len(detection_log)}")
        return detection_log
    
    def cleanup(self):
        """Cleanup camera resources and free memory"""
        try:
            if self.camera:
                self.camera.stop()
                self.camera.close()
                self.camera = None
                self.logger.info("Camera cleaned up")
            
            if self.model:
                del self.model
                self.model = None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                self.logger.info("AI model unloaded")
                
        except Exception as e:
            self.logger.error(f"Camera cleanup error: {e}")


# =============================================================================
# STANDALONE TEST FUNCTION
# =============================================================================

def test_ai_camera():
    """Test AI camera functionality"""
    print("🎥 Testing AI Camera Module")
    print("=" * 50)
    
    camera = AICamera()
    
    if camera.camera and camera.model:
        print("✅ Camera and AI model initialized")
        
        # Test regular photo capture
        print("\n📸 Testing photo capture...")
        photo_path = camera.capture_photo()
        if photo_path:
            print(f"✅ Photo saved: {photo_path}")
        
        # Test AI detection
        print("\n🤖 Testing AI threat detection...")
        detections, frame = camera.detect_threats()
        
        if detections:
            print(f"🚨 {len(detections)} threats detected:")
            for det in detections:
                print(f"  • {det['type']} (confidence: {det['confidence']:.2f}, "
                      f"level: {det['threat_level']})")
            
            # Save detection image
            if frame is not None:
                image_path = camera.save_detection_image(frame, detections)
                print(f"✅ Detection image saved: {image_path}")
        else:
            print("✅ No threats detected")
        
        # Show statistics
        print("\n📊 Detection Statistics:")
        stats = camera.get_detection_statistics()
        for key, value in stats.items():
            print(f"  • {key}: {value}")
    
    else:
        print("❌ Camera or AI model initialization failed")
    
    camera.cleanup()
    print("\n✅ Test complete")


if __name__ == "__main__":
    test_ai_camera()
