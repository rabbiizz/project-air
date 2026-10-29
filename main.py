import re
import time
import threading
import numpy as np
import cv2
from robot_control import MG400
import detector


# ============================================================
# [FIX 3] กล้องแบบ Thread แยก: อ่านภาพ + แสดงผล live ตลอดเวลา
# ภาพไม่ค้างแม้หุ่นยนต์กำลังเคลื่อนที่หรือโปรแกรมกำลัง sleep
# (imshow / waitKey ทั้งหมดอยู่ใน thread นี้ที่เดียว)
# มี read() เหมือน cv2.VideoCapture จึงส่งให้ detector.scan_objects ได้เลย
# ============================================================
class CameraStream:
    def __init__(self, src=0, window="Camera Live", center=(320, 240)):
        self.cap = cv2.VideoCapture(src)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.window = window
        self.center = center
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = False
        self.thread = None
        self.last_key = -1
        self.status = ""
        self.info = ""
        self.objects = []   # list ของ (name, cx, cy, rx, ry, color)

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
            cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if info:
                cv2.putText(disp, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.line(disp, (cx0 - 15, cy0), (cx0 + 15, cy0), (0, 165, 255), 2)
            cv2.line(disp, (cx0, cy0 - 15), (cx0, cy0 + 15), (0, 165, 255), 2)
            cv2.circle(disp, (cx0, cy0), 3, (0, 0, 255), -1)
            cv2.putText(disp, "Center", (cx0 + 10, cy0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            for (name, cx, cy, rx, ry, color) in objects:
                cv2.circle(disp, (cx, cy), 5, color, -1)
                cv2.putText(disp, f"{name}", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(disp, f"X:{rx:.1f} Y:{ry:.1f}", (cx - 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow(self.window, disp)
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
    HOME_POSE = [250.0, 0.0, 130.0, 0.0]  
    
    # ============================================================
    # [FIX 1] ตำแหน่งวาง (Slot)
    # !!! พิกัดเดิมของ Hexagon ห่างจาก Trapezoid แค่ ~2.5 mm และ Circle ห่างจาก Square แค่ ~5 mm
    # !!! หุ่นยนต์จึงดูเหมือน "วาง Hexagon ที่ช่อง Trapezoid" ทั้งที่โค้ดเลือกช่อง Hexagon ถูกแล้ว
    # !!! ให้ Teach ตำแหน่งช่อง Hexagon และ Circle จริงจาก Teach Pendant / DobotStudio แล้วใส่ค่าใหม่ตรงนี้
    # ============================================================
    SLOT_POSITIONS = {
        "Square":    [307.25, 130.65, -133.0, 0.0],  
        "Circle":    [350.03, 131.30, -133.0, 0.0],  
        "Trapezoid": [303.45, 70.872, -133.0, 0.0],  
        "Hexagon":   [349.44, 71.646, -133.0, 0.0]   
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

    # ============================================================
    # OFFSET ระหว่างพิกัดกล้องกับตำแหน่งหัวดูด (หน่วย: mm)
    # ============================================================
    OFFSET_X = 17.910 + 4
    OFFSET_Y = 11.509 - 3

    # ============================================================
    # [FIX 1] ค่าชดเชยตำแหน่งหัวดูดแยกตามรูปทรง (หน่วย: mm, แกนของหุ่นยนต์)
    # ใช้กรณีกล้องจัดศูนย์ตรงแล้ว แต่หัวดูดลงเยื้อง (เช่น Hexagon)
    # วิธีจูน: ดูค่า "Pick X/Y" ที่พิมพ์ใน Terminal เทียบกับตำแหน่งจริงของจุดกลางชิ้นงาน
    #          (Jog หัวดูดไปที่กลางชิ้นงานแล้วอ่านพิกัด) แล้วใส่ผลต่าง (จริง - Pick) ลงในช่องนี้
    # ============================================================
    PICK_CORRECTION = {
        "Square":    (0.0, 0.0),
        "Circle":    (0.0, 0.0),
        "Trapezoid": (0.0, 0.0),
        "Hexagon":   (0.0, 0.0),   # <-- ใส่ค่าชดเชยของ Hexagon ตรงนี้
    }

    # ฟังก์ชัน get_shape_params (สำหรับรักษามุมองศาของ Hexagon -16.97)
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

    # สมการ apply_offset แบบดั้งเดิมที่คำนวณตำแหน่งหัวดูดและองศาให้เสร็จสรรพ
    def apply_offset(shape_name, cam_x, cam_y):
        pt = np.array([[[cam_x, cam_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, align_matrix)

        camera_robot_x = float(transformed[0][0][0])
        camera_robot_y = float(transformed[0][0][1])

        target_x = camera_robot_x + OFFSET_X
        target_y = camera_robot_y + OFFSET_Y

        target_z, target_r = get_shape_params(shape_name)

        return target_x, target_y, target_z, target_r

    # ============================================================
    # [FIX 1] OFFSET หัวดูดหมุนตามมุม R ของหุ่นยนต์
    # กล้องติดอยู่บนแกนหมุน R เดียวกับหัวดูด เมื่อ R เปลี่ยน ตำแหน่งกล้องเทียบกับหัวดูดก็หมุนตาม
    # OFFSET_X/Y วัดไว้ที่ R = OFFSET_R_REF จึงต้องหมุนเวกเตอร์ OFFSET ตาม R ที่ใช้จริง
    # Hexagon ใช้ R = -16.97 ห่างจากมุมอ้างอิงมากที่สุด จึงเยื้องมากที่สุด
    # !!! ถ้าตอนวัด OFFSET หุ่นยนต์อยู่ที่ R อื่นที่ไม่ใช่ 0 ให้ใส่ค่านั้นแทน
    # ============================================================
    OFFSET_R_REF = 0.0

    def rotate_vec(vx, vy, r_deg):
        a = np.radians(r_deg - OFFSET_R_REF)
        return (vx * np.cos(a) - vy * np.sin(a),
                vx * np.sin(a) + vy * np.cos(a))

    def offset_at(r_deg):
        return rotate_vec(OFFSET_X, OFFSET_Y, r_deg)

    # ============================================================
    # [FIX 2] ตรวจพิกัดก่อนสั่งหุ่นยนต์ทุกครั้ง กัน Inverse kinematics error
    # ถ้าอยู่นอก Workspace จะไม่ส่งคำสั่ง และแจ้งว่าขั้นตอนไหน
    # ============================================================
    MIN_RADIUS = 150.0

    # ============================================================
    # [NEW] ตรวจสถานะหุ่นยนต์ และกู้คืนเมื่อเกิด Alarm / หลุด Enable
    # เดิมเมื่อหุ่นยนต์ Alarm โค้ดไม่รู้ ยังส่งคำสั่งต่อ (หุ่นไม่ทำตาม) แล้วจบโปรแกรมไปเลย
    # ตอนนี้: ก่อนส่งทุกคำสั่งจะเช็คสถานะ ถ้า Error/Disable -> Clear Error + Enable ใหม่ -> กลับ Home
    # รองรับทั้งเมธอดใน robot_control และคำสั่ง Dashboard ของ Dobot (RobotMode, ClearError, EnableRobot)
    # ============================================================
    MODE_DISABLED = 4
    MODE_ERROR = 9

    def _dashboard():
        for name in ("dashboard", "dash", "dashboard_api", "client_dash", "db", "dashboard_client"):
            d = getattr(robot, name, None)
            if d is not None:
                return d
        return None

    def _call_first(names, *args):
        """เรียกเมธอดแรกที่มีอยู่จริง ทั้งใน robot และ dashboard"""
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
        """Clear Error + Enable ใหม่ (ไม่ตัดการเชื่อมต่อ)"""
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
        """ถ้าหุ่นยนต์ Error หรือหลุด Enable ให้หยุดชิ้นนี้ (โยน Exception ให้ส่วนกู้คืนจัดการ)"""
        mode = robot_mode()
        if mode in (MODE_DISABLED, MODE_ERROR):
            raise RuntimeError(f"หุ่นยนต์ Error/หลุด Enable ก่อนขั้นตอน '{step}' "
                               f"(RobotMode = {mode}) {robot_error_text()}")

    # ============================================================
    # [NEW] รอให้หุ่นยนต์เคลื่อนที่ถึงที่หมายจริงก่อนทำขั้นต่อไป
    # robot.moveJ() แค่ส่งคำสั่งเข้าคิว แล้วโค้ดวิ่งต่อทันที -> โค้ดวิ่งนำหน้าหุ่นยนต์
    # พอถึงชิ้นสุดท้าย โค้ดจบและเรียก robot.close() ทั้งที่หุ่นยังทำคิวไม่เสร็จ
    # คำสั่งที่เหลือ (เช่น ลงหยิบ/วาง Circle) จึงถูกยกเลิก และหุ่นหลุด Enable
    # วิธีรอ (ใช้วิธีแรกที่ robot_control มี):
    #   1) Sync() / sync() ของ Dobot  2) อ่านตำแหน่งจริงจนถึงเป้า  3) อ่าน RobotMode จนไม่ใช่ Running
    #   4) ถ้าไม่มีเลย ประมาณเวลาจากระยะทาง (MOTION_MM_PER_S)
    # ============================================================
    MODE_RUNNING = 7
    MOTION_MM_PER_S = 60.0     # ความเร็วโดยประมาณที่ speed 30% (ใช้เฉพาะวิธีที่ 4)
    MOTION_TIMEOUT = 20.0      # วินาที รอได้นานสุดต่อ 1 คำสั่ง
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
            time.sleep(0.3)   # ให้หุ่นเริ่มวิ่งก่อน
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
        move_and_wait(x, y, z, r)   # [NEW] รอให้ถึงจริงก่อนไปขั้นต่อไป
        check_robot(step)           # [NEW] เช็คว่าไม่เกิด Alarm ระหว่างเคลื่อนที่

    # [FIX 3] รอเฟรมใหม่จาก Thread กล้อง เพื่อให้ภาพตรงกับตำแหน่งหุ่นยนต์ปัจจุบัน
    def flush_camera(cam, n=5):
        cam.wait_fresh(n)

    # กลับ Home อย่างปลอดภัย (ใช้ทั้งตอนจบงานและตอนเกิด Error)
    def go_home():
        try:
            # [NEW] ถ้าหุ่นยนต์ Error/หลุด Enable ให้กู้คืนก่อน แล้วค่อยกลับ Home
            if robot_mode() in (MODE_DISABLED, MODE_ERROR):
                recover_robot("(ก่อนกลับ Home)")
            elif hasattr(robot, "clear_error"):
                robot.clear_error()
            move_and_wait(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3])   # [NEW] รอถึง Home จริง
            time.sleep(0.5)
        except Exception as e:
            print(f"[ERROR] กลับ Home ไม่สำเร็จ: {e}")

    try:
        move_and_wait(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3])   # [NEW] รอถึง Home จริง
        time.sleep(0.5)
    except Exception as e:
        print(f"[ERROR] หุ่นยนต์ไม่สามารถเคลื่อนที่ได้ (ตรวจสอบ Limit): {e}")
        robot.close()
        return

    h_matrix = detector.load_calibration()
    cap = CameraStream(0, center=(IMG_CENTER_X, IMG_CENTER_Y))   # [FIX 3] กล้องแบบ Thread
    if not cap.isOpened():
        print("[ERROR] ไม่สามารถเปิดกล้องได้")
        robot.close()
        return
    cap.start()

    # [FIX 3] สร้างรายการจุดที่จะวาดทับภาพ (Thread กล้องเป็นคนวาด)
    def build_overlay(objs, color=(0, 0, 255)):
        items = []
        for o in objs:
            rx, ry, _, _ = apply_offset(o['name'], o['x'], o['y'])
            items.append((o['name'], o['cx'], o['cy'], rx, ry, color))
        return items

    print("\n[INFO] เปิดหน้าต่างกล้อง... กำลังดึงข้อมูลจาก detector มาแสดงผล")
    print(">>> กดปุ่ม 's' บนคีย์บอร์ด (ขณะคลิกที่หน้าต่างกล้อง) เพื่อเริ่มหยิบวัตถุ")
    print(">>> กดปุ่ม 'q' เพื่อยกเลิกและปิดโปรแกรม\n")

    detected_objects = []

    # ============================================================
    # การ Scan ครั้งแรกจาก Home Pose
    # ============================================================
    # ============================================================
    # [NEW] จำค่า x, y ตั้งแต่ตอนดูก่อนกด S
    # เก็บผลของ detector ย้อนหลัง REMEMBER_FRAMES เฟรม แล้วตอนกด S
    # ใช้ค่ากลาง (median) ของแต่ละชิ้น ค่าจึงนิ่ง ไม่ขึ้นกับเฟรมเดียวตอนกดปุ่ม
    # ============================================================
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
                # [NEW] เก็บมุมเอียงจาก detector (ค่ากลางของหลายเฟรม) ไว้ใช้หมุนก่อนวาง
                'angle': float(np.median([float(i.get('angle', 0.0)) for i in sel])),
            })
        return result

    cap.set_overlay(status="Alignment: 4-Point Perspective Active", info="")
    while True:
        detected_objects = detector.scan_objects(cap, h_matrix)
        cap.set_overlay(objects=build_overlay(detected_objects))
        scan_history.append(detected_objects)
        if len(scan_history) > REMEMBER_FRAMES:
            scan_history.pop(0)

        key = cap.pop_key()
        if key == ord('s'):
            print("[INFO] เริ่มกระบวนการหยิบวัตถุ...")
            # [NEW] ใช้ค่าที่จำไว้จากหลายเฟรมก่อนกด S
            detected_objects = remember_objects(scan_history)
            print(f"[INFO] จำพิกัดจาก {len(scan_history)} เฟรมก่อนกด S แล้ว")
            # [NEW] หลังกด S กล้อง live มีไว้ดูอย่างเดียว
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

    # คาดคะเนพิกัดเป้าหมายและตำแหน่งกล้องเริ่มต้น
    for obj in detected_objects:
        obj['img_angle'] = float(obj.get('angle', 0.0))   # [NEW] มุมเอียงจาก detector (เก็บก่อนถูกแทนด้วย R หยิบ)
        rx, ry, rz, rr = apply_offset(obj['name'], obj['x'], obj['y'])
        obj['x'] = rx
        obj['y'] = ry
        obj['z_target'] = -120.0
        obj['angle'] = rr

        # คำนวณหาพิกัดที่ต้องให้ "กล้อง" เลื่อนไปเล็งก่อน (โดยการหัก Offset หัวดูดออก)
        # [FIX 1] ใช้ OFFSET ที่หมุนตาม R ของรูปทรงนั้น
        off_x, off_y = offset_at(rr)
        obj['cam_target_x'] = rx - off_x
        obj['cam_target_y'] = ry - off_y

    # เรียงลำดับการหยิบ
    for obj in detected_objects:
        obj['distance'] = np.sqrt((obj['cx'] - IMG_CENTER_X)**2 + (obj['cy'] - IMG_CENTER_Y)**2)
    detected_objects = sorted(detected_objects, key=lambda k: k['distance'])
    
    print(f"[INFO] พบวัตถุทั้งหมด {len(detected_objects)} ชิ้น เรียงลำดับการหยิบเรียบร้อย")
    for o in detected_objects:
        print(f"       - {o['name']}  (X:{o['x']:.1f}, Y:{o['y']:.1f}, มุมเอียงจาก detector:{o['img_angle']:.1f}°)")

    # ============================================================
    # ลูปการทำงานทีละชิ้น (Visual Servoing & Pick/Place)
    # ============================================================
    Z_HOVER = 130
    Z_PICK = -135.5
    Z_PICK_3 = Z_PICK + 30
    STABLE_FRAMES = 3   # ต้องตรงศูนย์ติดกันกี่เฟรมจึงถือว่าตรงจริง

    # [FIX 2] ขีดจำกัดของการจัดศูนย์ กันหุ่นยนต์วิ่งไกลจนออกนอก Workspace
    SERVO_MAX_STEP = 5.0     # mm ขยับได้มากสุดต่อรอบ
    SERVO_MAX_DRIFT = 40.0   # mm ห่างจากจุดเริ่มได้มากสุด (เกินนี้ = จับผิดชิ้น/ภาพผิด)
    SERVO_MAX_ITERS = 60     # จำนวนรอบที่ขยับได้มากสุด
    SERVO_MAX_MISSES = 30    # [NEW] หาไม่เจอติดกันกี่เฟรมจึงยอมแพ้ (เดิมเฟรมเดียวก็ยอมแพ้)

    # ============================================================
    # [NEW] หมุนชิ้นงานก่อนวาง โดยใช้มุมเอียงจาก detector
    # ชิ้นงานบนโต๊ะเอียงอยู่ = มุมจาก detector -> แปลงเป็นมุมในแกนหุ่นยนต์ (ANGLE_SIGN)
    # ตอนหยิบ หัวดูดอยู่ที่ R หยิบ -> ก่อนวางหมุนหัวดูดเพื่อแก้มุมเอียงออก ให้ชิ้นงานตรงกับช่อง
    #   R วาง = R ของ Slot + R หยิบ - มุมเอียง(แกนหุ่นยนต์) + PLACE_ANGLE_OFFSET ของรูปทรง
    # รูปทรงสมมาตรจะเลือกมุมที่หมุนน้อยที่สุด (Square ทุก 90°, Hexagon ทุก 60°, Circle ไม่ต้องหมุน)
    # !!! ถ้าทดสอบแล้วชิ้นงานหมุนผิดทิศ (เอียงมากขึ้นแทนที่จะตรง) ให้เปลี่ยน ANGLE_SIGN เป็น +1.0
    # ============================================================
    ANGLE_SIGN = -1.0
    SYMMETRY_DEG = {"Square": 90.0, "Hexagon": 60.0, "Trapezoid": 360.0, "Circle": 0.0}   # [FIX] Trapezoid ต้องแยกด้านยาวบน/ล่าง จึงใช้ 360
    PLACE_ANGLE_OFFSET = {     # องศา ปรับละเอียดแยกรูปทรง ถ้าช่องวางเอียงไม่เท่ากัน
        "Square":    0.0,
        "Circle":    0.0,
        "Trapezoid": 0.0,
        "Hexagon":   0.0,
    }
    R_LIMIT = 160.0            # จำกัดมุม R ไม่ให้เกินช่วงหมุนของหุ่นยนต์
    ROTATE_DEG_PER_S = 60.0    # ความเร็วหมุนโดยประมาณ ใช้คำนวณเวลารอหมุนเสร็จ

    def place_angle(shape_name, img_angle, r_pick, slot_r):
        sym = SYMMETRY_DEG.get(shape_name, 180.0)
        if sym <= 0:
            tilt = 0.0                                   # Circle ไม่ต้องหมุน
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

        # กำหนดค่าเริ่มต้นของพิกัดหยิบจากการสแกนครั้งแรกของ "ชิ้นนี้"
        target_pick_x = obj['x']
        target_pick_y = obj['y']

        # [FIX 2] ครอบการทำงานแต่ละชิ้นด้วย try/except
        # ถ้าหุ่นยนต์ Error ระหว่างทาง (เช่นตอนหยิบ Circle) โปรแกรมจะไม่ดับทันที
        # แต่จะพิมพ์สาเหตุ ยกหัวขึ้น กลับ Home แล้วทำชิ้นถัดไปต่อ
        try:
            print(f"\n--- กำลังหยิบชิ้นที่ {idx+1}: {shape_name} ---")
            print(f"[INFO] เลื่อนกล้องไปตำแหน่งเริ่มต้น (Z=130)...")
            
            # เลื่อนหุ่นยนต์ให้กล้องไปอยู่ตรงเป้าหมายเริ่มต้น
            cap.set_overlay(status=f"Moving camera to {shape_name}", info="", objects=[])
            safe_moveJ("กล้องไปเหนือชิ้นงาน", cam_target_x, cam_target_y, Z_HOVER, target_angle)
            time.sleep(0.8)
            flush_camera(cap)

            # ============================================================
            # [NEW] Match center จากค่า x, y ที่จำไว้ตั้งแต่ก่อนกด S
            # ไม่ใช้ภาพจากกล้องหลังกด S มาคำนวณ (กล้อง live หลังกด S มีไว้ดูอย่างเดียว)
            # กล้องเลื่อนมาอยู่ตรงพิกัดที่จำไว้แล้ว -> ตำแหน่งหัวดูด = ตำแหน่งกล้อง + OFFSET (หมุนตาม R)
            # ============================================================
            print(f"[INFO] Match center จากพิกัดที่จำไว้ก่อนกด S")
            cap.set_overlay(status=f"Match center (stored X,Y): {shape_name}",
                            info=f"Stored X:{obj['x']:.1f} Y:{obj['y']:.1f}")
            off_x, off_y = offset_at(target_angle)
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

            # ลงไปหยิบชิ้นงานที่ Z = -120 ตามที่ต้องการ
            safe_moveJ("ลงหยิบ", target_pick_x, target_pick_y, Z_PICK, target_angle)
            time.sleep(0.3)
            check_robot("เปิดหัวดูด")
            robot.suction_pick()      # [NEW] ดูด: DO2 Pressure OFF -> DO1 Vacuum ON
            time.sleep(1.0)

            # ยกหัวดูดขึ้นกลับไปที่ Z=130
            print(f"[INFO] ยกชิ้นงาน {shape_name} ขึ้น")
            safe_moveJ("ยกชิ้นงานขึ้น", target_pick_x, target_pick_y, Z_HOVER, target_angle)
            time.sleep(0.5)

            # เคลื่อนที่ไปวางที่ Slot
            if shape_name in SLOT_POSITIONS:
                slot_x, slot_y, slot_z, slot_r = SLOT_POSITIONS[shape_name]
                print(f"[INFO] ไปวาง {shape_name} ที่ Slot X={slot_x:.2f}, Y={slot_y:.2f}")
                cap.set_overlay(status=f"Placing {shape_name}")

                safe_moveJ("ไปเหนือ Slot", slot_x, slot_y, Z_HOVER, slot_r)
                time.sleep(1.0)

                # [NEW] หมุนชิ้นงานเหนือช่องวางก่อนลงวาง ตามมุมเอียงจาก detector
                r_place, tilt = place_angle(shape_name, obj['img_angle'], target_angle, slot_r)
                print(f"[INFO] หมุน {shape_name} ก่อนวาง: มุมเอียงจาก detector {obj['img_angle']:.1f}° "
                      f"(แก้ {tilt:.1f}°) -> R วาง = {r_place:.2f}")
                cap.set_overlay(status=f"Rotating {shape_name} -> R {r_place:.1f}")
                safe_moveJ("หมุนชิ้นงานก่อนวาง", slot_x, slot_y, Z_HOVER, r_place)
                time.sleep(0.5 + abs(r_place - slot_r) / ROTATE_DEG_PER_S)   # รอหมุนเสร็จ แล้วหยุดนิ่งก่อนลง

                safe_moveJ("ลงวางที่ Slot", slot_x, slot_y, slot_z, r_place)
                time.sleep(0.5)

                check_robot("ปล่อยชิ้นงาน")
                robot.suction_release(blow_time=0.5)   # [NEW] ปล่อย: DO1 Vacuum OFF -> DO2 Pressure เป่า 0.5 วินาที -> OFF
                time.sleep(1.0)

                # ยกขึ้นเล็กน้อยหลังจากปล่อย
                safe_moveJ("ยกขึ้นจาก Slot", slot_x, slot_y, Z_HOVER, r_place)
                time.sleep(0.3)
            else:
                print(f"[WARNING] ไม่มีตำแหน่ง Slot ของ {shape_name}")

        except Exception as e:
            print(f"[ERROR] เกิดข้อผิดพลาดขณะทำงานกับ {shape_name}: {e}")
            print(f"[ERROR] ตำแหน่งล่าสุดที่สั่ง: Pick X={target_pick_x:.2f}, Y={target_pick_y:.2f}, R={target_angle:.2f}")
            try:
                # [NEW] Clear Error + Enable ใหม่ (ไม่ตัดการเชื่อมต่อ) แล้วยกหัวดูดขึ้นตรงๆ ก่อนกลับ Home
                if robot_mode() in (MODE_DISABLED, MODE_ERROR):
                    recover_robot()
                elif hasattr(robot, "clear_error"):
                    robot.clear_error()
                move_and_wait(target_pick_x, target_pick_y, Z_HOVER, target_angle)   # [NEW]
                time.sleep(0.5)
            except Exception as e2:
                print(f"[ERROR] ยกหัวดูดขึ้นไม่สำเร็จ: {e2}")

        # กลับ Home หลังทำเสร็จ 1 ชิ้น
        cap.set_overlay(status="Returning Home", info="", objects=[])
        go_home()   
        flush_camera(cap)

    # ============================================================
    # [FIX 3] วางครบทุกชิ้นแล้ว กลับ Home Position ก่อนปิดระบบ
    # ============================================================
    print("\n[INFO] วางครบทุกชิ้นแล้ว กำลังกลับ Home Position...")
    go_home()

    # [NEW] ตัดการเชื่อมต่อหลังจากหุ่นยนต์อยู่ที่ Home และหยุดนิ่งแล้วเท่านั้น
    wait_motion(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2])
    time.sleep(1.0)
    cap.release()
    robot.close()
    print("[INFO] ทำงานครบทุกชิ้น กลับ Home และปิดระบบเรียบร้อยครับ")

if __name__ == "__main__":
    main()