import cv2
import numpy as np
import os

def load_calibration():
    if os.path.exists('homography_matrix.npy'):
        h_matrix = np.load('homography_matrix.npy')
        print("โหลดไฟล์ค่าคาลิเบตสำเร็จ!")
        return h_matrix
    else:
        print("⚠️ ไม่พบไฟล์ 'homography_matrix.npy' กรุณารันไฟล์ calibrate.py ก่อน!")
        return None

def get_shape_name(approx, contour):
    shape = "Unknown"
    vertices = len(approx)

    if vertices == 4:
        pts = approx.reshape(4, 2)
        side_lengths = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            length = np.linalg.norm(p1 - p2)
            side_lengths.append(length)
        
        side_lengths.sort()
        min_side = side_lengths[0]
        max_side = side_lengths[3]
        ratio = min_side / max_side
        
        if ratio >= 0.78: 
            shape = "Square"
        else:
            shape = "Trapezoid"
    elif vertices == 5: 
        shape = "Unknown"
    elif vertices == 6:
        shape = "Hexagon"
    else:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.92 or vertices >= 7: 
                shape = "Circle"

    return shape

def get_shape_angle(cnt, shape_name):
    if shape_name == "Circle":
        return 0.0
    
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), raw_angle = rect
    
    if w < h:
        angle = raw_angle + 90
    else:
        angle = raw_angle
        
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
        
    return round(angle, 1)

def scan_objects(cap, h_matrix):
    """ฟังก์ชันสแกนหาวัตถุทั้งหมดจากเฟรมปัจจุบัน"""
    ret, frame = cap.read()
    if not ret:
        return []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])

    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            shape_name = get_shape_name(approx, cnt)

            if shape_name in ["Circle", "Square", "Trapezoid", "Hexagon"]:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                angle = get_shape_angle(cnt, shape_name)

                if h_matrix is not None:
                    pt = np.array([[[cX, cY]]], dtype=np.float32)
                    transformed = cv2.perspectiveTransform(pt, h_matrix)
                    real_coord = transformed[0][0]
                    posX_mm = real_coord[0]
                    posY_mm = real_coord[1]
                else:
                    posX_mm, posY_mm = float(cX), float(cY)

                detected_objects.append({
                    "name": shape_name,
                    "x": posX_mm,
                    "y": posY_mm,
                    "cx": cX,
                    "cy": cY,
                    "angle": angle
                })

    return detected_objects