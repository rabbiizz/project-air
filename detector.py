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

# ============================================================
# [NEW] มุมอ้างอิงของช่องวาง (องศา, แกนภาพกล้อง: 0° = ขวา, +90° = ลง)
# มุมที่ส่งให้ main = มุมที่ชิ้นงานเอียงไปจากทิศของช่องวาง (0 = หันตรงกับช่องแล้ว)
# ภาพจากกล้องบนหุ่นกลับหัว 180° เทียบกับรูปที่มองจากหน้าโต๊ะ
#   Hexagon  : ช่องวางยอดแหลมชี้ขึ้น-ลง -> ในภาพกล้องยังยอดแหลมชี้ขึ้น-ลง (มุมยอด -90° = 30° เมื่อหาร 60)
#   Trapezoid: ช่องวางด้านยาวอยู่บน     -> ในภาพกล้องด้านยาวอยู่ล่าง (ทิศด้านยาว = +90°)
# ถ้าทดสอบแล้ววางกลับด้าน ให้ปรับค่าตรงนี้ (Hexagon ทีละ 30, Trapezoid ทีละ 90 หรือ 180)
# ============================================================
HEX_REF_DEG = 30.0
TRAP_REF_DEG = 90.0


def _wrap(angle, period):
    """ห่อมุมให้อยู่ในช่วง [-period/2, period/2)"""
    return (angle + period / 2.0) % period - period / 2.0


def _edge_direction(cnt, p1, p2):
    """ทิศของด้านระหว่างยอด p1-p2 โดย fit เส้นตรงกับจุด contour ช่วงกลางของด้านนั้น (แม่นกว่าใช้ยอดตรงๆ)"""
    c = cnt.reshape(-1, 2).astype(np.float64)
    d = p2 - p1
    L = np.linalg.norm(d)
    if L < 1e-6:
        return np.arctan2(d[1], d[0])
    u = d / L
    t = (c - p1) @ u                                  # ตำแหน่งตามแนวด้าน
    dist = np.abs((c - p1) @ np.array([-u[1], u[0]]))  # ระยะห่างจากแนวด้าน
    sel = c[(t > 0.2 * L) & (t < 0.8 * L) & (dist < 0.08 * L + 2)]
    if len(sel) >= 5:
        vx, vy, _, _ = cv2.fitLine(sel.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        a = np.arctan2(vy, vx)
        if np.cos(a - np.arctan2(u[1], u[0])) < 0:   # ให้ทิศเดียวกับ p1 -> p2
            a += np.pi
        return a
    return np.arctan2(u[1], u[0])


def _hexagon_angle(cnt, approx, cx, cy):
    """มุมของหกเหลี่ยมจากทิศของด้านทั้ง 6 (หกเหลี่ยมหมุนครบรอบทุก 60°)"""
    pts = approx.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    s = c = 0.0
    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        a = _edge_direction(cnt, p1, p2)
        w = np.linalg.norm(p2 - p1)
        s += w * np.sin(6 * a)
        c += w * np.cos(6 * a)
    # ค่าเฉลี่ยเชิงวงกลมของ 6*ทิศด้าน -> ทิศของหกเหลี่ยม (คาบ 60°)
    # ยอดแหลมชี้ขึ้น-ลง: ด้านซ้าย/ขวาเป็นแนวตั้ง (90°) -> 30° เมื่อหาร 60 (เท่ากับ HEX_REF_DEG)
    mean6 = np.degrees(np.arctan2(s, c)) / 6.0
    return _wrap(mean6 - HEX_REF_DEG, 60.0)


def _trapezoid_angle(cnt, approx, cx, cy):
    """มุมของสี่เหลี่ยมคางหมูจากทิศของด้านยาว (หมุนครบรอบ 360° เพื่อแยกด้านยาวบน/ล่าง)"""
    pts = approx.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    lengths = [np.linalg.norm(pts[(i + 1) % n] - pts[i]) for i in range(n)]
    i = int(np.argmax(lengths))
    p1, p2 = pts[i], pts[(i + 1) % n]
    a = _edge_direction(cnt, p1, p2)
    if n == 4:
        # เฉลี่ยกับด้านสั้นที่ขนานกัน (ด้านตรงข้าม) ให้ทิศแม่นขึ้น
        q1, q2 = pts[(i + 2) % n], pts[(i + 3) % n]
        b = _edge_direction(cnt, q2, q1)              # กลับทิศให้ชี้ทางเดียวกับด้านยาว
        a = np.arctan2(np.sin(a) + np.sin(b), np.cos(a) + np.cos(b))
    normal = np.array([np.sin(a), -np.cos(a)])        # เส้นตั้งฉากกับด้านยาว
    mid = (p1 + p2) / 2.0
    if normal @ (mid - np.array([cx, cy])) < 0:       # ให้ชี้ออกจากจุดศูนย์กลางไปทางด้านยาว
        normal = -normal
    direction = np.degrees(np.arctan2(normal[1], normal[0]))
    return _wrap(direction - TRAP_REF_DEG, 360.0)


def get_shape_angle(cnt, shape_name, approx=None):
    if shape_name == "Circle":
        return 0.0

    # [NEW] Hexagon / Trapezoid ใช้ยอดของรูปทรงจริงแทน minAreaRect
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


# ============================================================
# [NEW] ถ้าเจอวงกลมกับสี่เหลี่ยมที่มั่นใจสูงแล้ว ชิ้นที่เหลือต้องเป็นหกเหลี่ยมหรือสี่เหลี่ยมคางหมู
#   ความมั่นใจวงกลม   : พื้นที่ / พื้นที่วงกลมที่ล้อมรูปพอดี (วงกลมจริง ~0.95-1.0, หกเหลี่ยม ~0.83)
#   ความมั่นใจสี่เหลี่ยม : มี 4 มุม, ด้านยาวเท่ากัน, เต็มกรอบสี่เหลี่ยมที่ล้อมรูป (คางหมูเต็มกรอบน้อยกว่า)
#   แยกหกเหลี่ยม/คางหมู : หกเหลี่ยมเต็มวงกลมที่ล้อมรูป ~0.83, คางหมู ~0.4-0.6
# ============================================================
CIRCLE_CONF_TH = 0.88      # ความมั่นใจขั้นต่ำของวงกลม (0-1)
SQUARE_CONF_TH = 0.85      # ความมั่นใจขั้นต่ำของสี่เหลี่ยม (0-1)
HEX_TRAP_FILL_TH = 0.68    # เต็มวงกลมที่ล้อมรูปมากกว่านี้ = หกเหลี่ยม, น้อยกว่า = คางหมู


def _circle_fill(cnt):
    """พื้นที่รูปทรง / พื้นที่วงกลมที่ล้อมรูปพอดี"""
    (_, _), r = cv2.minEnclosingCircle(cnt)
    return cv2.contourArea(cnt) / (np.pi * r * r) if r > 0 else 0.0


def _circle_confidence(cnt):
    return _circle_fill(cnt)


def _square_confidence(cnt):
    """ความมั่นใจว่าเป็นสี่เหลี่ยมจัตุรัส (0-1)"""
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
    """หา approxPolyDP ที่ได้จำนวนยอดเท่ากับ n (ใช้คำนวณมุมของรูปทรงที่ถูกแก้ชื่อ)"""
    peri = cv2.arcLength(cnt, True)
    for eps in np.linspace(0.01, 0.08, 15):
        ap = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(ap) == n:
            return ap
    return default


def _apply_circle_square_rule(candidates):
    """candidates: list ของ dict {cnt, approx, name} -> แก้ชื่อตามกฎวงกลม/สี่เหลี่ยม"""
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
        # 2 ชิ้นที่เหลือ: ชิ้นที่เต็มวงกลมที่ล้อมรูปมากกว่า = หกเหลี่ยม อีกชิ้น = คางหมู
        rest.sort(key=lambda i: _circle_fill(candidates[i]['cnt']), reverse=True)
        candidates[rest[0]]['name'] = "Hexagon"
        candidates[rest[1]]['name'] = "Trapezoid"
    else:
        for i in rest:
            fill = _circle_fill(candidates[i]['cnt'])
            candidates[i]['name'] = "Hexagon" if fill >= HEX_TRAP_FILL_TH else "Trapezoid"

    # ใช้ยอดที่ตรงกับรูปทรงใหม่ เพื่อให้คำนวณมุมได้ถูก
    for i in rest:
        n = 6 if candidates[i]['name'] == "Hexagon" else 4
        candidates[i]['approx'] = _approx_with_vertices(candidates[i]['cnt'], n, candidates[i]['approx'])
    return candidates


# ============================================================
# [NEW] รูปทรงที่ detect ไปแล้ว ห้ามซ้ำกับชิ้นอื่น (แต่ละรูปทรงมีได้ชิ้นเดียวในเฟรม)
# ถ้ามีชื่อซ้ำกัน จะจับคู่ชิ้นงานกับรูปทรงใหม่ทั้งหมดแบบไม่ให้ซ้ำ
# โดยเลือกแบบที่คะแนนรวมสูงสุด (ชื่อเดิมที่ detect ได้จะได้คะแนนพิเศษ จึงถูกเปลี่ยนเฉพาะชิ้นที่มั่นใจน้อย)
# ============================================================
UNIQUE_SHAPES = ["Circle", "Square", "Hexagon", "Trapezoid"]
KEEP_NAME_BONUS = 0.2      # คะแนนพิเศษถ้ารูปทรงตรงกับที่ detect ได้เดิม


def _shape_scores(cnt):
    """คะแนน (0-1) ว่าชิ้นงานนี้น่าจะเป็นแต่ละรูปทรงแค่ไหน"""
    fill = _circle_fill(cnt)
    return {
        "Circle":    _circle_confidence(cnt),
        "Square":    _square_confidence(cnt),
        "Hexagon":   max(0.0, 1.0 - abs(fill - 0.83) / 0.25),
        "Trapezoid": max(0.0, 1.0 - abs(fill - 0.50) / 0.30),
    }


def _apply_unique_rule(candidates):
    """ถ้ามีรูปทรงซ้ำกัน ให้จับคู่ใหม่ไม่ให้ซ้ำ ชิ้นที่เกินจำนวนรูปทรงจะเป็น Unknown"""
    names = [c['name'] for c in candidates if c['name'] in UNIQUE_SHAPES]
    if len(names) == len(set(names)):
        return candidates                                  # ไม่มีชื่อซ้ำ ไม่ต้องแก้

    scores = [_shape_scores(c['cnt']) for c in candidates]
    n = len(candidates)
    best_total, best_assign = -1.0, None
    k = min(n, len(UNIQUE_SHAPES))
    for idxs in itertools.permutations(range(n), k):       # เลือกชิ้นงาน k ชิ้นมาจับคู่กับรูปทรงที่ต่างกัน
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


def scan_objects(cap, h_matrix):
    """ฟังก์ชันสแกนหาวัตถุทั้งหมดจากเฟรมปัจจุบัน"""
    ret, frame = cap.read()
    if not ret:
        return []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 30, 0])
    upper_black = np.array([180, 255, 50])

    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []

    # [NEW] แยกรูปทรงทุกชิ้นก่อน (รวมชิ้นที่ยังเป็น Unknown) แล้วค่อยใช้กฎวงกลม/สี่เหลี่ยม
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            candidates.append({'cnt': cnt, 'approx': approx, 'name': get_shape_name(approx, cnt)})
    candidates = _apply_circle_square_rule(candidates)
    candidates = _apply_unique_rule(candidates)   # [NEW] รูปทรงที่เจอแล้วห้ามซ้ำ

    for cand in candidates:
        cnt, approx, shape_name = cand['cnt'], cand['approx'], cand['name']
        if shape_name in ["Circle", "Square", "Trapezoid", "Hexagon"]:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            angle = get_shape_angle(cnt, shape_name, approx)   # [NEW] ส่งยอดของรูปทรงไปด้วย

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