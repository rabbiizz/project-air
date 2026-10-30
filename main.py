import re
import time
import threading
import numpy as np
import cv2
from robot_control import MG400
import detector


# ============================================================
# กล้องแบบ Thread แยก: อ่านภาพ + แสดงผล live ตลอดเวลา
# ภาพไม่ค้างแม้หุ่นยนต์กำลังเคลื่อนที่หรือโปรแกรมกำลัง sleep
# (imshow / waitKey ทั้งหมดอยู่ใน thread นี้ที่เดียว)
# มี read() เหมือน cv2.VideoCapture จึงส่งให้ detector.scan_objects ได้เลย
# ============================================================
class CameraStream:
    def __init__(self, src=0, window="Window 1: Original + Shapes", center=(320, 240), h_matrix=None):
        # --- [แก้ไขจุดที่ 1] ระบุ Backend ให้ชัดเจนเพื่อความเร็ว ---
        # ถ้าใช้ Windows แนะนำให้ใช้ cv2.CAP_DSHOW
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        
        # (ถ้าใช้งานบน Linux ให้เปลี่ยนเป็น cv2.CAP_V4L2 แทนครับ)
        # self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # --- [แก้ไขจุดที่ 2] กำหนดขนาดภาพ (Resolution) ตั้งแต่เริ่มต้น ---
            # ช่วยลดเวลาที่กล้องต้องเสียไปกับการเคลียร์ค่าความละเอียดอัตโนมัติ
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.window = window
        self.center = center
        self.h_matrix = h_matrix
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = False
        self.thread = None
        self.last_key = -1
        self.status = ""
        self.info = ""
        self.objects = []   # list ของข้อมูลวัตถุ

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 5.0:
            time.sleep(0.01)
        return self

    def _loop(self):
        cx0, cy0 = self.center
        while self.running:
            ret, f = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f
                self.frame_id += 1
                status, info, objects = self.status, self.info, list(self.objects)

            disp = f.copy()

            # วาดกรอบการทำงาน 180x180 มม. (จากเซ็นเตอร์กล้องเป็นจุดกึ่งกลางขยายออกไป 180x180 มม.)
            center_px = np.array([[[float(cx0), float(cy0)]]], dtype=np.float32)
            if self.h_matrix is not None:
                try:
                    center_world = cv2.perspectiveTransform(center_px, self.h_matrix)[0][0]
                    c_x, c_y = center_world[0], center_world[1]
                    half_size = 150.0  # ขยายออกจากศูนย์กลางด้านละ 90 มม. (รวมเป็น 180 มม.)
                    box_world = np.array([[
                        [c_x - half_size, c_y - half_size],
                        [c_x + half_size, c_y - half_size],
                        [c_x + half_size, c_y + half_size],
                        [c_x - half_size, c_y + half_size]
                    ]], dtype=np.float32)
                    inv_h = np.linalg.inv(self.h_matrix)
                    box_px = cv2.perspectiveTransform(box_world, inv_h)
                    pts_box = box_px[0].astype(np.int32)
                    cv2.polylines(disp, [pts_box], isClosed=True, color=(0, 0, 255), thickness=2)
                except Exception:
                    pass
            else:
                half_px = 150
                pt1 = (int(cx0 - half_px), int(cy0 - half_px))
                pt2 = (int(cx0 + half_px), int(cy0 + half_px))
                cv2.rectangle(disp, pt1, pt2, (0, 0, 255), 2)

            cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if info:
                cv2.putText(disp, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.line(disp, (cx0 - 15, cy0), (cx0 + 15, cy0), (0, 165, 255), 2)
            cv2.line(disp, (cx0, cy0 - 15), (cx0, cy0 + 15), (0, 165, 255), 2)
            cv2.circle(disp, (cx0, cy0), 3, (0, 0, 255), -1)
            cv2.putText(disp, "Center", (cx0 + 10, cy0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            for item in objects:
                name = item[0]
                cx = item[1]
                cy = item[2]
                rx = item[3]
                ry = item[4]
                color = item[5]
                approx = item[6] if len(item) > 6 else None

                # วาดเส้นกรอบรอบรูปทรง
                if approx is not None:
                    cv2.polylines(disp, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

                cv2.circle(disp, (cx, cy), 5, color, -1)
                cv2.putText(disp, f"{name}", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(disp, f"X:{rx:.1f} Y:{ry:.1f}", (cx - 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # แสดงหน้าต่างหลัก Original + Shapes
            cv2.imshow(self.window, disp)

            # แสดงหน้าต่าง HSV Black Mask
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 80])
            hsv_mask = cv2.inRange(hsv, lower_black, upper_black)
            hsv_mask[380:, :] = 0  # ตัดขอบล่างออกเพื่อความสะอาด
            kernel = np.ones((5, 5), np.uint8)
            hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
            hsv_mask = cv2.dilate(hsv_mask, kernel, iterations=1)
            cv2.imshow("Window 2: HSV Black Mask", hsv_mask)

            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                with self.lock:
                    self.last_key = k
        cv2.destroyAllWindows()

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def grab(self):
        return True

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def set_overlay(self, status=None, info=None, objects=None):
        with self.lock:
            if status is not None:
                self.status = status
            if info is not None:
                self.info = info
            if objects is not None:
                self.objects = objects

    def pop_key(self):
        with self.lock:
            k = self.last_key
            self.last_key = -1
        return k

    def wait_fresh(self, n=3, timeout=2.0):
        """รอเฟรมใหม่ n เฟรม เพื่อให้ภาพตรงกับตำแหน่งหุ่นยนต์ปัจจุบัน"""
        start_id = self.frame_id
        t0 = time.time()
        while self.frame_id < start_id + n and time.time() - t0 < timeout:
            time.sleep(0.005)

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.cap.release()


def main():
    # 1. เชื่อมต่อหุ่นยนต์ MG400
    robot = MG400(ip="192.168.1.6")
    
    if not robot.use_real_robot:
        print("[ERROR] ไม่สามารถเชื่อมต่อกับหุ่นยนต์ได้")
        return
        
    print("[INFO] เชื่อมต่อและ Enable หุ่นยนต์สำเร็จ!")
    robot.set_speed(30)

    # กำหนดพิกัดจุด Home P5 [250, 0, 130, 0]
    HOME_POSE = [250.0, 0.0, 130.0, 50]  
    
    SLOT_POSITIONS = {
        "Square":    [307.31, 130.03, -132.0, 0.0],  
        "Circle":    [350.03, 131.30, -133.0, 0.0],  
        "Trapezoid": [304.66, 71.655, -131.5, 0.0],  
        "Hexagon":   [350.67, 71.368, -133.0, 0.0]   
    }

    MAX_REACH = 400.0
    MIN_SLOT_GAP = 30.0  # mm ช่องวางแต่ละช่องควรห่างกันอย่างน้อยเท่านี้

    # ตรวจสอบพิกัด Slot ก่อนเริ่มทำงาน
    slot_names = list(SLOT_POSITIONS.keys())
    for i in range(len(slot_names)):
        sx, sy = SLOT_POSITIONS[slot_names[i]][0], SLOT_POSITIONS[slot_names[i]][1]
        if np.hypot(sx, sy) > MAX_REACH:
            print(f"[WARNING] Slot {slot_names[i]} อยู่เกินระยะ {MAX_REACH} mm")
        for j in range(i + 1, len(slot_names)):
            tx, ty = SLOT_POSITIONS[slot_names[j]][0], SLOT_POSITIONS[slot_names[j]][1]
            gap = np.hypot(sx - tx, sy - ty)
            if gap < MIN_SLOT_GAP:
                print(f"[WARNING] Slot {slot_names[i]} กับ {slot_names[j]} ห่างกันแค่ {gap:.1f} mm "
                      f"(จะวางทับที่เดียวกัน) กรุณาแก้พิกัดใน SLOT_POSITIONS")

    IMG_CENTER_X, IMG_CENTER_Y = 320, 240

    # โหลดค่า Calibration สำหรับการตรวจจับรูปทรง
    h_matrix = detector.load_calibration()

    # --- จุดพิกัดแปลง 4 จุด ---
    pts_cam = np.array([
        [119.7, 21.0],    # P1 (Square)
        [121.2, 112.5],   # P2 (Hexagon)
        [12.5, 28.9],     # P3 (Trapezoid)
        [-7.0, 122.5]     # P4 (Circle)
    ], dtype=np.float32)

    pts_rob = np.array([
        [314.79, -66.27], # P1
        [225.58, -68.77], # P2
        [305.07, 33.942], # P3
        [215.59, 53.030]  # P4
    ], dtype=np.float32)

    align_matrix = cv2.getPerspectiveTransform(pts_cam, pts_rob)
    print("[INFO] สร้างเมทริกซ์แปลงพิกัด (4-Point Alignment) สำเร็จ")

    # [NEW] คำนวณจากตำแหน่งหัวดูดจริงที่กลางชิ้นงาน (C, S, H, T) เทียบกับค่าที่กล้องคำนวณได้
    #       ค่าเฉลี่ยผลต่าง dX = -1.130, dY = -1.908 -> 17.910 + 8.5 - 1.130 = 25.280, 11.509 - 8.5 - 1.908 = 1.101
    # [NEW] ปรับจากตำแหน่งหัวดูดจริง P6-P9 (S, H, T, C) เทียบกับค่าที่กล้องคำนวณได้บนจอ
    #       ค่าเฉลี่ยผลต่าง dX = +0.373, dY = -1.051
    OFFSET_X = 25.280 - 7.5
    OFFSET_Y = 1.101 + 1

    # ============================================================
    # [NEW] Match center: หลังกด S กล้องไปที่พิกัดที่จำไว้ แล้วปรับจนจุดกลางวัตถุตรงกลางภาพ (Error <= 1 px)
    # จุดลงหยิบใช้สูตรเดิม: พิกัดที่จำไว้ก่อนกด S + OFFSET_X / OFFSET_Y (หมุนตาม R)
    # ============================================================
    SERVO_TOL_PX = 1          # px ถือว่าตรงเมื่อ Error ทั้ง X และ Y ไม่เกินนี้
    SERVO_KP = 0.15           # mm ต่อ 1 px
    SERVO_STABLE_FRAMES = 3   # ต้องตรงติดกันกี่ครั้งจึงถือว่าตรงจริง
    SERVO_MAX_STEP = 5.0      # mm ขยับได้มากสุดต่อรอบ
    SERVO_MAX_DRIFT = 40.0    # mm ห่างจากจุดเริ่มได้มากสุด (เกิน = จับผิดชิ้น)
    SERVO_MAX_ITERS = 60      # จำนวนรอบที่ขยับได้มากสุด
    SERVO_MAX_MISSES = 30     # หาชิ้นงานไม่เจอติดกันกี่ครั้งจึงยอมแพ้
    SERVO_MATCH_RADIUS_PX = 200   # [NEW] รับเฉพาะชิ้นงานที่อยู่ห่างกลางภาพไม่เกินนี้ (กันจับถาด/ชิ้นอื่นที่ขอบภาพ)

    PICK_CORRECTION = {
        "Square":    (0.0, 0.0),
        "Circle":    (0.0, 0.0),
        "Trapezoid": (0.0, 0.0),
        "Hexagon":   (0.0, 0.0),
    }

    def get_shape_params(shape_name):
        if shape_name == "Square":      # P1
            return -137.2, -11.90
        elif shape_name == "Hexagon":   # P2
            return -134.1, -16.97
        elif shape_name == "Trapezoid": # P3
            return -136.5, 6.3346
        elif shape_name == "Circle":    # P4
            return -133.9, 13.800
        return 0.0, 0.0

    def apply_offset(shape_name, cam_x, cam_y):
        pt = np.array([[[cam_x, cam_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, align_matrix)

        camera_robot_x = float(transformed[0][0][0])
        camera_robot_y = float(transformed[0][0][1])

        target_x = camera_robot_x + OFFSET_X
        target_y = camera_robot_y + OFFSET_Y

        target_z, target_r = get_shape_params(shape_name)

        return target_x, target_y, target_z, target_r

    OFFSET_R_REF = 0.0

    def rotate_vec(vx, vy, r_deg):
        a = np.radians(r_deg - OFFSET_R_REF)
        return (vx * np.cos(a) - vy * np.sin(a),
                vx * np.sin(a) + vy * np.cos(a))

    def offset_at(r_deg):
        return rotate_vec(OFFSET_X, OFFSET_Y, r_deg)

    # ============================================================
    # [NEW] ระยะจาก "ตำแหน่งหุ่น" ไปยัง "จุดหยิบของวัตถุที่อยู่กลางภาพพอดี" (CAM_CENTER_X / Y)
    #   OFFSET_X / Y ใช้กับพิกัดที่แปลงจากภาพตอนอยู่ Home (align + OFFSET = จุดหยิบ)
    #   แต่ไม่ใช่ระยะจากกลางภาพไปหัวดูด จึงคำนวณจากค่าคาลิเบรตเดิม:
    #   ตอนอยู่ Home วัตถุที่อยู่กลางภาพ (320,240) มีจุดหยิบ = align(h(กลางภาพ)) + OFFSET
    #   -> CAM_CENTER = จุดหยิบนั้น - HOME (หมุนตาม R เหมือน OFFSET)
    #   -> หลัง Match center: จุดหยิบ = ตำแหน่งกล้องตอนวัตถุอยู่กลางภาพ + CAM_CENTER
    # ============================================================
    def compute_cam_center_offset():
        if h_matrix is None:
            print("[WARNING] ไม่มี homography_matrix ใช้ OFFSET_X / OFFSET_Y เป็นระยะกล้องกลางภาพ-หัวดูด")
            return OFFSET_X, OFFSET_Y
        c_px = np.array([[[float(IMG_CENTER_X), float(IMG_CENTER_Y)]]], dtype=np.float32)
        c_w = cv2.perspectiveTransform(c_px, h_matrix)[0][0]
        pick_x, pick_y, _, _ = apply_offset("", float(c_w[0]), float(c_w[1]))
        return pick_x - HOME_POSE[0], pick_y - HOME_POSE[1]

    CAM_CENTER_X, CAM_CENTER_Y = compute_cam_center_offset()
    print(f"[INFO] ระยะกล้องกลางภาพ -> หัวดูด (ที่ R=0): X={CAM_CENTER_X:.2f}, Y={CAM_CENTER_Y:.2f} mm")

    def cam_center_at(r_deg):
        return rotate_vec(CAM_CENTER_X, CAM_CENTER_Y, r_deg)

    # [NEW] มุม R ตอนกล้องมอง/Match center (หมุนให้สายลมไม่บังกล้อง)
    #       ตอนหัวดูดลงหยิบยังใช้ R ของแต่ละรูปทรงจาก get_shape_params เหมือนเดิม
    VIEW_R = 50.0

    # [NEW] มุมที่ภาพกล้องหมุนไปจริง (ใช้หมุน CAM_CENTER และทิศการขยับตอน Match center)
    #       จากภาพตอน R = 50 ภาพไม่ได้หมุนตาม R (กล้องไม่ได้หมุนไปกับหัวดูด)
    #       กล้องหมุนตามแขนหุ่น (แกน J1 = atan2(Y, X)) เทียบกับตอนอยู่ Home ที่ใช้คาลิเบรต
    #       ถ้าเดิมใช้ R (เช่น 50) หมุนทิศ -> ขยับผิดทิศ หาจุดกลางไม่เจอแล้วกลับ Home
    CAM_FOLLOWS_R = False     # True = กล้องหมุนตาม R (แบบเดิม), False = กล้องหมุนตามแขน J1

    def cam_rot(x, y, r_deg):
        if CAM_FOLLOWS_R:
            return r_deg
        return float(np.degrees(np.arctan2(y, x)) - np.degrees(np.arctan2(HOME_POSE[1], HOME_POSE[0])))

    MIN_RADIUS = 150.0

    MODE_DISABLED = 4
    MODE_ERROR = 9

    def _dashboard():
        for name in ("dashboard", "dash", "dashboard_api", "client_dash", "db", "dashboard_client"):
            d = getattr(robot, name, None)
            if d is not None:
                return d
        return None

    def _call_first(names, *args):
        for obj in (robot, _dashboard()):
            if obj is None:
                continue
            for n in names:
                fn = getattr(obj, n, None)
                if callable(fn):
                    return True, fn(*args)
        return False, None

    def _parse_int(ret):
        if isinstance(ret, (int, np.integer)):
            return int(ret)
        if ret is None:
            return None
        s = str(ret)
        m = re.search(r"\{\s*(-?\d+)", s)
        if m:
            return int(m.group(1))
        m = re.search(r"-?\d+", s)
        return int(m.group(0)) if m else None

    def robot_mode():
        try:
            ok, ret = _call_first(("get_mode", "robot_mode", "get_robot_mode", "RobotMode"))
            return _parse_int(ret) if ok else None
        except Exception:
            return None

    def robot_error_text():
        try:
            ok, ret = _call_first(("get_error_id", "GetErrorID"))
            return str(ret) if ok else ""
        except Exception:
            return ""

    def recover_robot(reason=""):
        print(f"[RECOVER] หุ่นยนต์อยู่ในสถานะ Error/Disable {reason} {robot_error_text()}")
        try:
            _call_first(("clear_error", "ClearError"))
            time.sleep(0.5)
            _call_first(("enable", "enable_robot", "EnableRobot"))
            time.sleep(2.0)
            _call_first(("set_speed",), 30)
            print(f"[RECOVER] Clear Error และ Enable ใหม่แล้ว (RobotMode = {robot_mode()})")
        except Exception as e:
            print(f"[RECOVER] กู้คืนไม่สำเร็จ: {e}")

    def check_robot(step):
        mode = robot_mode()
        if mode in (MODE_DISABLED, MODE_ERROR):
            raise RuntimeError(f"หุ่นยนต์ Error/หลุด Enable ก่อนขั้นตอน '{step}' "
                               f"(RobotMode = {mode}) {robot_error_text()}")

    MODE_RUNNING = 7
    MOTION_MM_PER_S = 60.0     
    MOTION_TIMEOUT = 20.0      
    last_cmd = list(HOME_POSE)

    def _find_method(names):
        for obj in (robot, _dashboard(), getattr(robot, "move", None), getattr(robot, "move_api", None)):
            if obj is None:
                continue
            for n in names:
                fn = getattr(obj, n, None)
                if callable(fn):
                    return fn
        return None

    sync_fn = _find_method(("sync", "Sync", "wait_until_done", "wait_motion_done"))
    pose_fn = _find_method(("get_pose", "getPose", "get_current_pose", "GetPose"))

    def read_pose():
        if pose_fn is None:
            return None
        try:
            p = pose_fn()
            if isinstance(p, str):
                inner = re.search(r"\{([^}]*)\}", p)
                vals = [float(v) for v in re.findall(r"-?\d+\.?\d*", inner.group(1) if inner else p)]
            else:
                vals = [float(v) for v in list(p)]
            return vals[:4] if len(vals) >= 3 else None
        except Exception:
            return None

    def wait_motion(x, y, z):
        t0 = time.time()
        if sync_fn is not None:
            try:
                sync_fn()
                return
            except Exception:
                pass
        if pose_fn is not None and read_pose() is not None:
            while time.time() - t0 < MOTION_TIMEOUT:
                p = read_pose()
                if p is None:
                    break
                if np.hypot(p[0] - x, p[1] - y) < 0.5 and abs(p[2] - z) < 0.5:
                    time.sleep(0.1)
                    return
                if robot_mode() in (MODE_DISABLED, MODE_ERROR):
                    return
                time.sleep(0.05)
            print(f"[WARNING] รอหุ่นยนต์ถึงเป้าหมายเกิน {MOTION_TIMEOUT} วินาที")
            return
        if robot_mode() is not None:
            time.sleep(0.3)   
            while time.time() - t0 < MOTION_TIMEOUT and robot_mode() == MODE_RUNNING:
                time.sleep(0.05)
            return
        dist = np.sqrt((x - last_cmd[0])**2 + (y - last_cmd[1])**2 + (z - last_cmd[2])**2)
        time.sleep(0.5 + dist / MOTION_MM_PER_S)

    if sync_fn is not None:
        print("[INFO] รอการเคลื่อนที่ด้วย Sync()")
    elif pose_fn is not None:
        print("[INFO] รอการเคลื่อนที่ด้วยการอ่านตำแหน่งจริงของหุ่นยนต์")
    elif robot_mode() is not None:
        print("[INFO] รอการเคลื่อนที่ด้วย RobotMode")
    else:
        print(f"[WARNING] robot_control ไม่มีคำสั่งอ่านสถานะ ใช้การประมาณเวลาจากระยะทาง ({MOTION_MM_PER_S} mm/s)")

    def move_and_wait(x, y, z, r):
        robot.moveJ(x, y, z, r)
        wait_motion(x, y, z)
        last_cmd[:] = [x, y, z, r]

    def safe_moveJ(step, x, y, z, r):
        radius = np.hypot(x, y)
        if not np.isfinite([x, y, z, r]).all() or not (MIN_RADIUS <= radius <= MAX_REACH):
            raise RuntimeError(f"{step}: X={x:.1f} Y={y:.1f} (รัศมี {radius:.1f} mm) อยู่นอก Workspace "
                               f"ไม่ส่งคำสั่งให้หุ่นยนต์")
        check_robot(step)
        move_and_wait(x, y, z, r)   
        check_robot(step)           

    # [NEW] เคลื่อนที่แบบเส้นตรง (MovL) ใช้ตอนลง/ขึ้นที่ Slot
    # MovJ หมุนข้อต่อแบบเส้นตรงในปริภูมิข้อต่อ เมื่อลง Z ระยะไกลปลายแขนจะโค้งออกด้านข้าง (แนวรัศมี)
    # ทำให้ชิ้นงานที่ถืออยู่ไปชนชิ้นที่วางไว้แล้วใน Slot ข้างเคียง -> ใช้ MovL ให้ลงตรงแนวดิ่ง
    def safe_moveL(step, x, y, z, r):
        radius = np.hypot(x, y)
        if not np.isfinite([x, y, z, r]).all() or not (MIN_RADIUS <= radius <= MAX_REACH):
            raise RuntimeError(f"{step}: X={x:.1f} Y={y:.1f} (รัศมี {radius:.1f} mm) อยู่นอก Workspace "
                               f"ไม่ส่งคำสั่งให้หุ่นยนต์")
        check_robot(step)
        robot.moveL(x, y, z, r)
        wait_motion(x, y, z)
        last_cmd[:] = [x, y, z, r]
        check_robot(step)

    def flush_camera(cam, n=5):
        cam.wait_fresh(n)

    def go_home():
        try:
            if robot_mode() in (MODE_DISABLED, MODE_ERROR):
                recover_robot("(ก่อนกลับ Home)")
            elif hasattr(robot, "clear_error"):
                robot.clear_error()
            move_and_wait(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3])   
            time.sleep(0.5)
        except Exception as e:
            print(f"[ERROR] กลับ Home ไม่สำเร็จ: {e}")

    try:
        move_and_wait(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3])   
        time.sleep(0.5)
    except Exception as e:
        print(f"[ERROR] หุ่นยนต์ไม่สามารถเคลื่อนที่ได้ (ตรวจสอบ Limit): {e}")
        robot.close()
        return

    # เปิดสตรีมกล้อง (ส่ง h_matrix เข้าไปเพื่อใช้วาดกรอบการทำงาน 180x180 มม.)
    cap = CameraStream(1, center=(IMG_CENTER_X, IMG_CENTER_Y), h_matrix=h_matrix)
    if not cap.isOpened():
        print("[ERROR] ไม่สามารถเปิดกล้องได้")
        robot.close()
        return
    cap.start()

    def build_overlay(objs, color=(0, 0, 255)):
        items = []
        for o in objs:
            rx, ry, _, _ = apply_offset(o['name'], o['x'], o['y'])
            items.append((o['name'], o['cx'], o['cy'], rx, ry, color, o.get('approx')))
        return items

    print("\n[INFO] เปิดหน้าต่างกล้อง... กำลังดึงข้อมูลจาก detector มาแสดงผล")
    print(">>> กดปุ่ม 's' บนคีย์บอร์ด (ขณะคลิกที่หน้าต่างกล้อง) เพื่อเริ่มหยิบวัตถุ")
    print(">>> กดปุ่ม 'q' เพื่อยกเลิกและปิดโปรแกรม\n")

    detected_objects = []

    REMEMBER_FRAMES = 15
    scan_history = []

    def remember_objects(history, radius_px=25):
        clusters = []
        for dets in history:
            for d in dets:
                for c in clusters:
                    if np.hypot(d['cx'] - c['cx'], d['cy'] - c['cy']) < radius_px:
                        c['items'].append(d)
                        break
                else:
                    clusters.append({'cx': d['cx'], 'cy': d['cy'], 'items': [d]})
        min_votes = max(1, len(history) // 2)
        result = []
        for c in clusters:
            items = c['items']
            if len(items) < min_votes:
                continue
            names = [i['name'] for i in items]
            best = max(set(names), key=names.count)
            sel = [i for i in items if i['name'] == best]
            result.append({
                'name': best,
                'x': float(np.median([i['x'] for i in sel])),
                'y': float(np.median([i['y'] for i in sel])),
                'cx': int(round(np.median([i['cx'] for i in sel]))),
                'cy': int(round(np.median([i['cy'] for i in sel]))),
                'angle': float(np.median([float(i.get('angle', 0.0)) for i in sel])),
                'approx': sel[0].get('approx') if sel else None
            })
        return result

    cap.set_overlay(status="Shape Detection Active", info="")
    while True:
        detected_objects = detector.scan_objects(cap, h_matrix)
        cap.set_overlay(objects=build_overlay(detected_objects))
        scan_history.append(detected_objects)
        if len(scan_history) > REMEMBER_FRAMES:
            scan_history.pop(0)

        key = cap.pop_key()
        if key == ord('s'):
            print("[INFO] เริ่มกระบวนการหยิบวัตถุ...")
            detected_objects = remember_objects(scan_history)
            print(f"[INFO] จำพิกัดจาก {len(scan_history)} เฟรมก่อนกด S แล้ว")
            cap.set_overlay(status="Live view (stored X,Y in use)", info="", objects=[])
            break
        elif key == ord('q'):
            print("[INFO] ยกเลิกการทำงาน")
            cap.release()
            robot.close()
            return
        time.sleep(0.03)

    if not detected_objects:
        print("[INFO] ไม่พบวัตถุใดๆ ในพื้นที่")
        cap.release()
        robot.close()
        return

    for obj in detected_objects:
        obj['img_angle'] = float(obj.get('angle', 0.0))   
        rx, ry, rz, rr = apply_offset(obj['name'], obj['x'], obj['y'])
        obj['x'] = rx
        obj['y'] = ry
        obj['z_target'] = -120.0
        obj['angle'] = rr

        # [NEW] ตำแหน่งที่มองครั้งแรก -> ตำแหน่งกล้องที่ทำให้วัตถุอยู่กลางภาพ (จุดหยิบที่จำไว้ - CAM_CENTER)
        # [NEW] หมุน CAM_CENTER ตามมุมที่ภาพกล้องหมุนจริงที่ตำแหน่งกล้อง (คำนวณซ้ำ 2 รอบให้ลู่เข้า)
        c_x, c_y = rx, ry
        for _ in range(2):
            off_x, off_y = cam_center_at(cam_rot(c_x, c_y, VIEW_R))
            c_x, c_y = rx - off_x, ry - off_y
        obj['cam_target_x'] = c_x
        obj['cam_target_y'] = c_y

    for obj in detected_objects:
        obj['distance'] = np.sqrt((obj['cx'] - IMG_CENTER_X)**2 + (obj['cy'] - IMG_CENTER_Y)**2)
    detected_objects = sorted(detected_objects, key=lambda k: k['distance'])
    
    print(f"[INFO] พบวัตถุทั้งหมด {len(detected_objects)} ชิ้น เรียงลำดับการหยิบเรียบร้อย")
    for o in detected_objects:
        print(f"       - {o['name']}  (X:{o['x']:.1f}, Y:{o['y']:.1f}, มุมเอียงจาก detector:{o['img_angle']:.1f}°)")

    Z_HOVER = 130
    Z_PICK = -136
    Z_PICK_3 = Z_PICK + 30
    Z_PLACE_APPROACH = 30.0   # [NEW] mm หยุดเหนือ Slot ก่อนลงวางช้าๆ ตรงแนวดิ่ง

    ANGLE_SIGN = -1.0
    SYMMETRY_DEG = {"Square": 90.0, "Hexagon": 60.0, "Trapezoid": 360.0, "Circle": 0.0}   
    PLACE_ANGLE_OFFSET = {     
        "Square":    0.0,
        "Circle":    0.0,
        "Trapezoid": 0.0,
        "Hexagon":   0.0,
    }
    R_LIMIT = 160.0            
    ROTATE_DEG_PER_S = 60.0    

    def place_angle(shape_name, img_angle, r_pick, slot_r):
        sym = SYMMETRY_DEG.get(shape_name, 180.0)
        if sym <= 0:
            tilt = 0.0                                   
        else:
            tilt = (float(img_angle) + sym / 2.0) % sym - sym / 2.0
        tilt_robot = ANGLE_SIGN * tilt
        r = slot_r + r_pick - tilt_robot + PLACE_ANGLE_OFFSET.get(shape_name, 0.0)
        r = (r + 180.0) % 360.0 - 180.0
        return max(-R_LIMIT, min(R_LIMIT, r)), tilt

    for idx, obj in enumerate(detected_objects):
        shape_name = obj['name']
        cam_target_x = obj['cam_target_x']
        cam_target_y = obj['cam_target_y']
        target_angle = obj['angle']

        target_pick_x = obj['x']
        target_pick_y = obj['y']

        try:
            print(f"\n--- กำลังหยิบชิ้นที่ {idx+1}: {shape_name} ---")
            print(f"[INFO] เลื่อนกล้องไปตำแหน่งเริ่มต้น (Z=130)...")
            
            cap.set_overlay(status=f"Moving camera to {shape_name}", info="", objects=[])
            safe_moveJ("กล้องไปเหนือชิ้นงาน", cam_target_x, cam_target_y, Z_HOVER, target_angle)
            time.sleep(0.8)
            flush_camera(cap)

            # [NEW] ไปถึงตำแหน่งที่มองครั้งแรกแล้ว -> ลืมตำแหน่งที่จำไว้ของชิ้นนี้
            #       (จุดหยิบจะมาจาก Match center เท่านั้น เก็บไว้แค่ชื่อรูปทรงและมุมสำหรับตอนวาง)
            for k in ('x', 'y', 'cx', 'cy', 'cam_target_x', 'cam_target_y', 'distance', 'approx'):
                obj.pop(k, None)
            target_pick_x, target_pick_y = cam_target_x, cam_target_y
            print(f"[INFO] ลืมตำแหน่งที่จำไว้ครั้งแรกของ {shape_name} แล้ว -> เริ่ม Match center "
                  f"ให้ Error <= {SERVO_TOL_PX} px")

            # [NEW] หมุน R เป็น VIEW_R (50) ก่อน Match center (อยู่กับที่ X, Y เดิม) ให้สายลมไม่บังกล้อง
            print(f"[INFO] หมุน R จาก {target_angle:.2f} เป็น {VIEW_R:.2f} ก่อน Match center")
            cap.set_overlay(status=f"Rotate R -> {VIEW_R:.1f} for match center", info="")
            safe_moveJ("หมุน R ก่อน Match center", cam_target_x, cam_target_y, Z_HOVER, VIEW_R)
            time.sleep(0.5 + abs(VIEW_R - target_angle) / ROTATE_DEG_PER_S)
            flush_camera(cap)

            # [NEW] Match center: ปรับกล้องจนจุดกลางวัตถุตรงกลางภาพ (Error <= 1 px)
            cap.set_overlay(status=f"Match center: {shape_name}", info="")
            start_cam_x, start_cam_y = cam_target_x, cam_target_y
            servo_ok = False
            iters = stable_count = miss_count = 0
            while True:
                current_objs = detector.scan_objects(cap, h_matrix)
                cap.set_overlay(objects=build_overlay(current_objs, color=(255, 0, 0)))
                matched_obj, min_d = None, float('inf')
                for c_obj in current_objs:
                    if c_obj['name'] == shape_name:
                        d = np.hypot(c_obj['cx'] - IMG_CENTER_X, c_obj['cy'] - IMG_CENTER_Y)
                        if d < min_d and d <= SERVO_MATCH_RADIUS_PX:   # ต้องอยู่ใกล้กลางภาพ (กันจับถาด/ชิ้นอื่น)
                            min_d, matched_obj = d, c_obj

                if matched_obj is None:
                    miss_count += 1
                    cap.set_overlay(info=f"Searching {shape_name}... ({miss_count}/{SERVO_MAX_MISSES})")
                    if miss_count >= SERVO_MAX_MISSES:
                        print(f"[WARNING] ไม่พบชิ้นงาน {shape_name} ติดกัน {miss_count} ครั้ง")
                        break
                    flush_camera(cap, 1)
                    continue
                miss_count = 0

                err_x = matched_obj['cx'] - IMG_CENTER_X
                err_y = matched_obj['cy'] - IMG_CENTER_Y
                cap.set_overlay(info=f"Error X:{err_x} Y:{err_y} px  iter:{iters}")

                if abs(err_x) <= SERVO_TOL_PX and abs(err_y) <= SERVO_TOL_PX:
                    stable_count += 1
                    if stable_count >= SERVO_STABLE_FRAMES:
                        print(f"[INFO] Match center สำเร็จ (Err X: {err_x} px, Err Y: {err_y} px, {iters} รอบ)")
                        servo_ok = True
                        break
                    flush_camera(cap, 1)
                    continue
                stable_count = 0

                if iters >= SERVO_MAX_ITERS:
                    print(f"[WARNING] Match center ไม่สำเร็จใน {iters} รอบ")
                    break

                move_x = -err_y * SERVO_KP
                move_y = -err_x * SERVO_KP
                move_x, move_y = rotate_vec(move_x, move_y, cam_rot(cam_target_x, cam_target_y, VIEW_R))   # [NEW] หมุนตามมุมภาพจริง
                step = np.hypot(move_x, move_y)
                if step > SERVO_MAX_STEP:
                    move_x *= SERVO_MAX_STEP / step
                    move_y *= SERVO_MAX_STEP / step
                new_x, new_y = cam_target_x + move_x, cam_target_y + move_y
                if np.hypot(new_x - start_cam_x, new_y - start_cam_y) > SERVO_MAX_DRIFT:
                    print(f"[WARNING] Match center เลื่อนเกิน {SERVO_MAX_DRIFT} mm (อาจจับผิดชิ้น)")
                    break
                cam_target_x, cam_target_y = new_x, new_y
                safe_moveJ("Match center", cam_target_x, cam_target_y, Z_HOVER, VIEW_R)   # [NEW] R = VIEW_R
                iters += 1
                time.sleep(0.3)
                flush_camera(cap)

            # ตำแหน่งที่จำไว้ถูกลืมแล้ว -> Match center ไม่สำเร็จ = ข้ามชิ้นนี้ (ไม่ลงหยิบแบบเดา)
            if not servo_ok:
                print(f"[WARNING] ข้าม {shape_name} เพราะ Match center ไม่สำเร็จ (ไม่มีตำแหน่งที่จำไว้ให้ใช้แทน)")
                cap.set_overlay(status=f"Skip {shape_name}", info="")
                cap.set_overlay(status="Returning Home", info="", objects=[])
                go_home()
                flush_camera(cap)
                continue

            # [NEW] Match center เสร็จ -> หมุน R กลับเป็นค่าเดิมที่คำนวณไว้ (อยู่กับที่ X, Y เดิม) แล้วค่อยลงหยิบ
            print(f"[INFO] หมุน R กลับจาก {VIEW_R:.2f} เป็นค่าเดิม {target_angle:.2f} ก่อนลงหยิบ")
            cap.set_overlay(status=f"Rotate R back -> {target_angle:.1f}", info="")
            safe_moveJ("หมุน R กลับค่าเดิม", cam_target_x, cam_target_y, Z_HOVER, target_angle)
            time.sleep(0.5 + abs(VIEW_R - target_angle) / ROTATE_DEG_PER_S)

            # จุดลงหยิบ = ตำแหน่งกล้องตอนวัตถุอยู่กลางภาพ + CAM_CENTER (หมุนตาม R)
            off_x, off_y = cam_center_at(cam_rot(cam_target_x, cam_target_y, VIEW_R))   # [NEW] มุมภาพจริงตอนวัตถุอยู่กลางภาพ
            corr_x, corr_y = PICK_CORRECTION.get(shape_name, (0.0, 0.0))
            target_pick_x = cam_target_x + off_x + corr_x
            target_pick_y = cam_target_y + off_y + corr_y

            cap.set_overlay(status=f"Picking {shape_name}", info="")
            print(f"[INFO] {shape_name}: Camera X={cam_target_x:.2f}, Y={cam_target_y:.2f} | "
                  f"Pick X={target_pick_x:.2f}, Y={target_pick_y:.2f}, R={target_angle:.2f}")
            safe_moveJ("หัวดูดไปเหนือชิ้นงาน", target_pick_x, target_pick_y, Z_HOVER, target_angle)
            time.sleep(0.5)

            safe_moveJ("ลดหัวดูดลง", target_pick_x, target_pick_y, Z_PICK_3, target_angle)
            time.sleep(0.3)

            safe_moveJ("ลงหยิบ", target_pick_x, target_pick_y, Z_PICK, target_angle)
            time.sleep(0.3)
            check_robot("เปิดหัวดูด")
            robot.suction_pick()      
            time.sleep(1.0)

            print(f"[INFO] ยกชิ้นงาน {shape_name} ขึ้น")
            # [NEW] ยกขึ้นตรงแนวแกน Z อย่างเดียวด้วย MovL (X, Y, R คงเดิม) ไม่ให้ชิ้นงานเหวี่ยง/ลากไปชนชิ้นข้างๆ
            safe_moveL("ยกชิ้นงานขึ้นเหนือจุดหยิบ", target_pick_x, target_pick_y, Z_PICK_3, target_angle)
            time.sleep(0.3)
            safe_moveL("ยกชิ้นงานขึ้น", target_pick_x, target_pick_y, Z_HOVER, target_angle)
            time.sleep(0.5)

            if shape_name in SLOT_POSITIONS:
                slot_x, slot_y, slot_z, slot_r = SLOT_POSITIONS[shape_name]
                print(f"[INFO] ไปวาง {shape_name} ที่ Slot X={slot_x:.2f}, Y={slot_y:.2f}")
                cap.set_overlay(status=f"Placing {shape_name}")

                safe_moveJ("ไปเหนือ Slot", slot_x, slot_y, Z_HOVER, slot_r)
                time.sleep(1.0)

                r_place, tilt = place_angle(shape_name, obj['img_angle'], target_angle, slot_r)
                print(f"[INFO] หมุน {shape_name} ก่อนวาง: มุมเอียงจาก detector {obj['img_angle']:.1f}° "
                      f"(แก้ {tilt:.1f}°) -> R วาง = {r_place:.2f}")
                cap.set_overlay(status=f"Rotating {shape_name} -> R {r_place:.1f}")
                safe_moveJ("หมุนชิ้นงานก่อนวาง", slot_x, slot_y, Z_HOVER, r_place)
                time.sleep(0.5 + abs(r_place - slot_r) / ROTATE_DEG_PER_S)   

                # [NEW] ลงตรงแนวดิ่งด้วย MovL: Z_HOVER -> เหนือ Slot 30 mm -> Slot
                #       กันชิ้นงานที่ถืออยู่เหวี่ยงไปชนชิ้นที่วางไว้แล้ว (Hexagon / Circle)
                safe_moveL("ลงเหนือ Slot", slot_x, slot_y, slot_z + Z_PLACE_APPROACH, r_place)
                time.sleep(0.3)
                safe_moveL("ลงวางที่ Slot", slot_x, slot_y, slot_z, r_place)
                time.sleep(0.5)

                check_robot("ปล่อยชิ้นงาน")
                robot.suction_release(blow_time=0.5)   
                time.sleep(1.0)

                # [NEW] ยกขึ้นตรงแนวดิ่งด้วย MovL ไม่ให้หัวดูดเฉี่ยวชิ้นที่วางไว้ข้างๆ
                safe_moveL("ยกขึ้นเหนือ Slot", slot_x, slot_y, slot_z + Z_PLACE_APPROACH, r_place)
                time.sleep(0.3)
                safe_moveL("ยกขึ้นจาก Slot", slot_x, slot_y, Z_HOVER, r_place)
                time.sleep(0.3)
            else:
                print(f"[WARNING] ไม่มีตำแหน่ง Slot ของ {shape_name}")

        except Exception as e:
            print(f"[ERROR] เกิดข้อผิดพลาดขณะทำงานกับ {shape_name}: {e}")
            print(f"[ERROR] ตำแหน่งล่าสุดที่สั่ง: Pick X={target_pick_x:.2f}, Y={target_pick_y:.2f}, R={target_angle:.2f}")
            try:
                if robot_mode() in (MODE_DISABLED, MODE_ERROR):
                    recover_robot()
                elif hasattr(robot, "clear_error"):
                    robot.clear_error()
                move_and_wait(target_pick_x, target_pick_y, Z_HOVER, target_angle)   
                time.sleep(0.5)
            except Exception as e2:
                print(f"[ERROR] ยกหัวดูดขึ้นไม่สำเร็จ: {e2}")

        cap.set_overlay(status="Returning Home", info="", objects=[])
        go_home()   
        flush_camera(cap)

    print("\n[INFO] วางครบทุกชิ้นแล้ว กำลังกลับ Home Position...")
    go_home()

    wait_motion(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2])
    time.sleep(1.0)
    cap.release()
    robot.close()
    print("[INFO] ทำงานครบทุกชิ้น กลับ Home และปิดระบบเรียบร้อยครับ")


if __name__ == "__main__":
    main()