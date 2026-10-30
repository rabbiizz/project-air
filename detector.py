import cv2
import numpy as np
import os
import itertools

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
    
    approx4 = None
    peri = cv2.arcLength(contour, True)
    for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
        ap = cv2.approxPolyDP(contour, eps * peri, True)
        if len(ap) == 4:
            approx4 = ap
            break
            
    if approx4 is None:
        approx4 = approx
        
    vertices = len(approx4)

    if vertices in (4, 5):
        pts = approx4.reshape(-1, 2)
        if len(pts) == 4:
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
        else:
            shape = "Trapezoid"
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

HEX_REF_DEG = 30.0
TRAP_REF_DEG = 90.0

def _wrap(angle, period):
    return (angle + period / 2.0) % period - period / 2.0

def _edge_direction(cnt, p1, p2):
    c = cnt.reshape(-1, 2).astype(np.float64)
    d = p2 - p1
    L = np.linalg.norm(d)
    if L < 1e-6:
        return np.arctan2(d[1], d[0])
    u = d / L
    t = (c - p1) @ u                                  
    dist = np.abs((c - p1) @ np.array([-u[1], u[0]]))  
    sel = c[(t > 0.2 * L) & (t < 0.8 * L) & (dist < 0.08 * L + 2)]
    if len(sel) >= 5:
        vx, vy, _, _ = cv2.fitLine(sel.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        a = np.arctan2(vy, vx)
        if np.cos(a - np.arctan2(u[1], u[0])) < 0:   
            a += np.pi
        return a
    return np.arctan2(u[1], u[0])

def _hexagon_angle(cnt, approx, cx, cy):
    pts = approx.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    s = c = 0.0
    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        a = _edge_direction(cnt, p1, p2)
        w = np.linalg.norm(p2 - p1)
        s += w * np.sin(6 * a)
        c += w * np.cos(6 * a)
    mean6 = np.degrees(np.arctan2(s, c)) / 6.0
    return _wrap(mean6 - HEX_REF_DEG, 60.0)

def _trapezoid_angle(cnt, approx, cx, cy):
    pts = approx.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    lengths = [np.linalg.norm(pts[(i + 1) % n] - pts[i]) for i in range(n)]
    i = int(np.argmax(lengths))
    p1, p2 = pts[i], pts[(i + 1) % n]
    a = _edge_direction(cnt, p1, p2)
    if n == 4:
        q1, q2 = pts[(i + 2) % n], pts[(i + 3) % n]
        b = _edge_direction(cnt, q2, q1)              
        a = np.arctan2(np.sin(a) + np.sin(b), np.cos(a) + np.cos(b))
    normal = np.array([np.sin(a), -np.cos(a)])        
    mid = (p1 + p2) / 2.0
    if normal @ (mid - np.array([cx, cy])) < 0:       
        normal = -normal
    direction = np.degrees(np.arctan2(normal[1], normal[0]))
    return _wrap(direction - TRAP_REF_DEG, 360.0)

def get_shape_angle(cnt, shape_name, approx=None):
    if shape_name == "Circle":
        return 0.0

    if approx is not None and shape_name in ("Hexagon", "Trapezoid"):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            if shape_name == "Hexagon":
                return round(float(_hexagon_angle(cnt, approx, cx, cy)), 1)
            return round(float(_trapezoid_angle(cnt, approx, cx, cy)), 1)

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

CIRCLE_CONF_TH = 0.88      
SQUARE_CONF_TH = 0.85      
HEX_TRAP_FILL_TH = 0.70    

def _circle_fill(cnt):
    (_, _), r = cv2.minEnclosingCircle(cnt)
    return cv2.contourArea(cnt) / (np.pi * r * r) if r > 0 else 0.0

def _circle_confidence(cnt):
    return _circle_fill(cnt)

def _square_confidence(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    if len(approx) != 4:
        return 0.0
    pts = approx.reshape(4, 2).astype(np.float64)
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    side_ratio = min(sides) / max(sides) if max(sides) > 0 else 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(cnt)
    rect_fill = cv2.contourArea(cnt) / (w * h) if w * h > 0 else 0.0
    aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
    return min(side_ratio, rect_fill, aspect)

def _approx_with_vertices(cnt, n, default):
    peri = cv2.arcLength(cnt, True)
    for eps in np.linspace(0.01, 0.08, 15):
        ap = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(ap) == n:
            return ap
    return default

def _apply_circle_square_rule(candidates):
    if len(candidates) < 3:
        return candidates
    circ = [(_circle_confidence(c['cnt']), i) for i, c in enumerate(candidates)]
    best_c = max(circ)
    if best_c[0] < CIRCLE_CONF_TH:
        return candidates
    sq = [(_square_confidence(c['cnt']), i) for i, c in enumerate(candidates) if i != best_c[1]]
    best_s = max(sq) if sq else (0.0, -1)
    if best_s[0] < SQUARE_CONF_TH:
        return candidates

    candidates[best_c[1]]['name'] = "Circle"
    candidates[best_s[1]]['name'] = "Square"
    rest = [i for i in range(len(candidates)) if i not in (best_c[1], best_s[1])]

    if len(rest) == 2:
        rest.sort(key=lambda i: _circle_fill(candidates[i]['cnt']), reverse=True)
        candidates[rest[0]]['name'] = "Hexagon"
        candidates[rest[1]]['name'] = "Trapezoid"
    else:
        for i in rest:
            fill = _circle_fill(candidates[i]['cnt'])
            candidates[i]['name'] = "Hexagon" if fill >= HEX_TRAP_FILL_TH else "Trapezoid"

    for i in rest:
        n = 6 if candidates[i]['name'] == "Hexagon" else 4
        candidates[i]['approx'] = _approx_with_vertices(candidates[i]['cnt'], n, candidates[i]['approx'])
    return candidates

UNIQUE_SHAPES = ["Circle", "Square", "Hexagon", "Trapezoid"]
KEEP_NAME_BONUS = 0.2      

def _shape_scores(cnt):
    fill = _circle_fill(cnt)
    return {
        "Circle":    _circle_confidence(cnt),
        "Square":    _square_confidence(cnt),
        "Hexagon":   max(0.0, 1.0 - abs(fill - 0.83) / 0.25),
        "Trapezoid": max(0.0, 1.0 - abs(fill - 0.50) / 0.30),
    }

def _apply_unique_rule(candidates):
    names = [c['name'] for c in candidates if c['name'] in UNIQUE_SHAPES]
    # แก้ไข: ต้องมีครบทั้ง 4 รูปทรงแบบไม่ซ้ำกันจริงๆ ถึงจะปล่อยผ่าน (หากมี Unknown จะบังคับรันจัดสรรใหม่ทันที)
    if len(names) == len(UNIQUE_SHAPES) and len(set(names)) == len(UNIQUE_SHAPES):
        return candidates                                  

    scores = [_shape_scores(c['cnt']) for c in candidates]
    n = len(candidates)
    best_total, best_assign = -1.0, None
    k = min(n, len(UNIQUE_SHAPES))
    for idxs in itertools.permutations(range(n), k):       
        for shapes in itertools.permutations(UNIQUE_SHAPES, k):
            total = 0.0
            for i, sh in zip(idxs, shapes):
                total += scores[i][sh] + (KEEP_NAME_BONUS if candidates[i]['name'] == sh else 0.0)
            if total > best_total:
                best_total, best_assign = total, dict(zip(idxs, shapes))

    vertices = {"Hexagon": 6, "Trapezoid": 4, "Square": 4}
    for i, c in enumerate(candidates):
        new_name = best_assign.get(i, "Unknown")
        if new_name != c['name'] and new_name in vertices:
            c['approx'] = _approx_with_vertices(c['cnt'], vertices[new_name], c['approx'])
        c['name'] = new_name
    return candidates

# ============================================================
# [NEW] กันไม่ให้ถาดวาง (สีน้ำเงินเข้ม) ถูกตรวจจับเป็นชิ้นงาน
# วัดจากภาพกล้อง: ถาด H 106-119, S 73-116, V 61-78 | ชิ้นงานสีดำ V 19-40, S ต่ำกว่า
#   1) ตัดพิกเซลสีน้ำเงินของถาดออกจาก mask ก่อนหา contour
#   2) ไม่รับ contour ที่ใหญ่เกินขนาดชิ้นงาน (ถาดใหญ่กว่าชิ้นงานหลายเท่า)
# ถ้ายังมีส่วนของถาดหลุดเข้ามา ให้ลด TRAY_S_MIN / TRAY_V_MIN ลง
# ============================================================
TRAY_H_MIN, TRAY_H_MAX = 95, 135   # ช่วงสีน้ำเงิน (Hue ของ OpenCV 0-180)
TRAY_S_MIN = 55                    # ความอิ่มสีขั้นต่ำของถาด
TRAY_V_MIN = 45                    # ความสว่างขั้นต่ำของถาด
MAX_OBJECT_AREA = 30000            # px^2 contour ที่ใหญ่กว่านี้ไม่ใช่ชิ้นงาน

# ============================================================
# [NEW] ไม่นับชิ้นงานที่วางอยู่ในถาดแล้ว
#   ปัญหา: ตอนหยิบชิ้นสุดท้าย ชิ้นที่วางในถาดแล้ว (สีดำ) ยังถูกนับเป็น candidate
#          ชิ้นในถาดถูกขอบถาดบังจนรูปแหว่ง (fill ต่ำ) -> แย่งชื่อ "Trapezoid" ไปในกฎห้ามซ้ำ
#          คางหมูตัวจริงถูกเปลี่ยนชื่อ -> main.py หา "Trapezoid" ไม่เจอตอน Match center
#   แก้: หาพื้นที่ถาด (ก้อนสีน้ำเงินใหญ่ -> convex hull) แล้วตัด contour ที่อยู่ในถาดออก
#        ก่อนเข้ากฎ circle/square และกฎห้ามซ้ำ
# ============================================================
TRAY_MIN_AREA = 3000               # px^2 ก้อนสีน้ำเงินที่ใหญ่กว่านี้ถือเป็นถาด
TRAY_MARGIN_PX = 10                # px ขยายขอบพื้นที่ถาดออกเล็กน้อย
TRAY_OVERLAP_TH = 0.30             # contour ที่ทับพื้นที่ถาดเกินสัดส่วนนี้ = ชิ้นในถาด ไม่นับ


def _tray_region(tray_mask):
    """คืน mask พื้นที่ถาด (convex hull ของก้อนสีน้ำเงินใหญ่ รวมชิ้นงานที่วางอยู่ข้างใน)"""
    region = np.zeros_like(tray_mask)
    cnts, _ = cv2.findContours(tray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) >= TRAY_MIN_AREA:
            cv2.fillPoly(region, [cv2.convexHull(c)], 255)
    if TRAY_MARGIN_PX > 0 and region.any():
        k = np.ones((2 * TRAY_MARGIN_PX + 1, 2 * TRAY_MARGIN_PX + 1), np.uint8)
        region = cv2.dilate(region, k, iterations=1)
    return region


def _in_tray(cnt, region):
    if not region.any():
        return False
    m = np.zeros_like(region)
    cv2.drawContours(m, [cnt], -1, 255, -1)
    total = cv2.countNonZero(m)
    if total == 0:
        return False
    overlap = cv2.countNonZero(cv2.bitwise_and(m, region))
    return overlap / total > TRAY_OVERLAP_TH

# ============================================================
# [NEW] วงกลม: fit วงรีด้วย cv2.fitEllipse แล้วขีดเส้นรอบเป็นรูปหลายเหลี่ยม CIRCLE_SEGMENTS เส้น (> 16)
#       ด้วย cv2.ellipse2Poly จากนั้นหา center จากรูปหลายเหลี่ยมนั้น
#       (ขอบหยัก/เงาไม่ทำให้ center เพี้ยนเหมือนใช้ contour ตรงๆ)
# ============================================================
CIRCLE_SEGMENTS = 24               # จำนวนเส้นรอบวงกลม (ต้องมากกว่า 16)


def _circle_polygon_center(cnt):
    """คืน (approx, cX, cY) ของวงกลมจากรูปหลายเหลี่ยม CIRCLE_SEGMENTS เส้น หรือ None ถ้า fit ไม่ได้"""
    if len(cnt) < 5:
        return None
    (ex, ey), (ma, mb), ang = cv2.fitEllipse(cnt)
    if ma <= 0 or mb <= 0:
        return None
    step = max(1, int(round(360.0 / CIRCLE_SEGMENTS)))
    poly = cv2.ellipse2Poly((int(round(ex)), int(round(ey))),
                            (int(round(ma / 2.0)), int(round(mb / 2.0))),
                            int(round(ang)), 0, 360, step)
    if len(poly) > 1 and np.array_equal(poly[0], poly[-1]):
        poly = poly[:-1]                       # ตัดจุดซ้ำปลายทางออก เหลือ CIRCLE_SEGMENTS จุด/เส้น
    if len(poly) <= 16:
        return None
    approx = poly.reshape(-1, 1, 2).astype(np.int32)
    M = cv2.moments(approx)
    if M["m00"] != 0:
        cX = int(round(M["m10"] / M["m00"]))
        cY = int(round(M["m01"] / M["m00"]))
    else:
        cX, cY = int(round(ex)), int(round(ey))
    return approx, cX, cY


def scan_objects(cap, h_matrix):
    ret, frame = cap.read()
    if not ret:
        return []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])

    mask = cv2.inRange(hsv, lower_black, upper_black)
    # [NEW] ตัดพิกเซลสีน้ำเงินของถาดออก
    tray_mask = cv2.inRange(hsv, np.array([TRAY_H_MIN, TRAY_S_MIN, TRAY_V_MIN]),
                            np.array([TRAY_H_MAX, 255, 255]))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(tray_mask))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []
    tray_region = _tray_region(tray_mask)   # [NEW] พื้นที่ถาด (ชิ้นที่วางแล้วอยู่ในนี้)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 1000 < area <= MAX_OBJECT_AREA:   # [NEW] ไม่รับ contour ที่ใหญ่เกินชิ้นงาน (เช่น ถาด)
            if _in_tray(cnt, tray_region):   # [NEW] ชิ้นที่วางในถาดแล้ว ไม่ให้เข้าไปแย่งชื่อรูปทรง
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            candidates.append({'cnt': cnt, 'approx': approx, 'name': get_shape_name(approx, cnt)})
            
    candidates = _apply_circle_square_rule(candidates)
    candidates = _apply_unique_rule(candidates)   

    for cand in candidates:
        cnt, approx, shape_name = cand['cnt'], cand['approx'], cand['name']
        if shape_name in ["Circle", "Square", "Trapezoid", "Hexagon"]:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # [NEW] วงกลม: ขีดเส้นรอบ CIRCLE_SEGMENTS เส้น (> 16) จากวงรีที่ fit ได้ แล้วหา center จากรูปนั้น
            if shape_name == "Circle":
                circ = _circle_polygon_center(cnt)
                if circ is not None:
                    approx, cX, cY = circ

            angle = get_shape_angle(cnt, shape_name, approx)   

            if h_matrix is not None:
                pt = np.array([[[cX, cY]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(pt, h_matrix)
                real_coord = transformed[0][0]
                posX_mm = real_coord[0]
                posY_mm = real_coord[1]
            else:
                posX_mm, posY_mm = float(cX), float(cY)

            center_px = np.array([[[320.0, 240.0]]], dtype=np.float32)
            if h_matrix is not None:
                center_world = cv2.perspectiveTransform(center_px, h_matrix)[0][0]
                c_x, c_y = center_world[0], center_world[1]
            else:
                c_x, c_y = 320.0, 240.0

            if abs(posX_mm - c_x) <= 150.0 and abs(posY_mm - c_y) <= 150.0:
                detected_objects.append({
                    "name": shape_name,
                    "x": posX_mm,
                    "y": posY_mm,
                    "cx": cX,
                    "cy": cY,
                    "angle": angle,
                    "approx": approx
                })

    return detected_objects