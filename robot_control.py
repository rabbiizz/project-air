import socket
import time


class MG400:
    def __init__(self, ip="192.168.1.6", dashboard_port=29999, motion_port=30003, use_real_robot=True, **kwargs):
        self.ip = ip
        self.dashboard_port = dashboard_port
        self.motion_port = motion_port
        self.use_real_robot = use_real_robot # เพิ่มบรรทัดนี้
        self.dashboard = None
        self.motion = None
        self.port = dashboard_port # กำหนด self.port เพื่อใช้ใน _prepare_robot

        if self.use_real_robot:
            try:
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client.connect((self.ip, self.port))
                self._prepare_robot()
                print(f"[INFO] Connected to MG400 at {self.ip}")
            except Exception as e:
                print(f"[ERROR] Connection failed: {e}")
                self.use_real_robot = False

    def set_speed(self, percentage):
        """ตั้งค่าความเร็วหุ่นยนต์ (1-100%)"""
        # ป้องกันการใส่ค่าเกินช่วงที่กำหนด
        percentage = max(1, min(percentage, 100))
        cmd = f"SpeedFactor({percentage})"
        print(f"[INFO] Setting robot speed to {percentage}%")
        return self.send_command(self.client, cmd)  # ส่งคำสั่งผ่านพอร์ต 29999 พร้อม \n[cite: 1]

    def _prepare_robot(self):
        self.send_command(self.client, "ClearError()")
        time.sleep(0.5)
        self.send_command(self.client, "EnableRobot()")
        time.sleep(2)

    def send_command(self, sock, cmd):
        if not cmd.endswith("\n"):
            cmd += "\n"  # ใช้ Newline ตามโปรโตคอล Dashboard[cite: 1]
        sock.send(cmd.encode("utf-8"))
        time.sleep(0.1)
        return sock.recv(1024).decode("utf-8")

    def moveL(self, x, y, z, r=0):
        """เคลื่อนที่แบบเส้นตรง (MovL)"""
        if self.use_real_robot:
            cmd = f"MovL({x},{y},{z},{r})"
            self.send_command(self.client, cmd)
        else:
            print(f"[MOCK MOVE L] X={x:.2f}, Y={y:.2f}, Z={z:.2f}, R={r:.2f}")

    # [NEW] เคลื่อนที่แบบ Joint (MovJ) ให้ main.py เรียกผ่าน robot.moveJ()
    # หุ่นคำนวณเส้นทางทีละแกนเอง ไม่บังคับให้วิ่งเป็นเส้นตรง จึงเลี่ยง Inverse kinematics error ระหว่างทางได้
    def moveJ(self, x, y, z, r=0):
        """เคลื่อนที่แบบ Joint (MovJ) ไปยังพิกัด X, Y, Z, R"""
        if self.use_real_robot:
            cmd = f"MovJ({x},{y},{z},{r})"
            self.send_command(self.client, cmd)
        else:
            print(f"[MOCK MOVE J] X={x:.2f}, Y={y:.2f}, Z={z:.2f}, R={r:.2f}")

    # เพิ่มใน class MG400 (source 2)
    def get_pose(self):
        """ดึงพิกัดปัจจุบัน X, Y, Z, R จากหุ่นยนต์"""
        if self.use_real_robot:
            res = self.send_command(self.client, "GetPose()")
            # ตัวอย่างตอบกลับ: "0,{320.5,10.2,-120.0,0.0}"
            try:
                # ทำความสะอาดสตริงเพื่อดึงตัวเลข
                pose_str = res.split("{")[1].split("}")[0]
                coords = [float(x) for x in pose_str.split(",")]
                return coords  # [X, Y, Z, R]
            except Exception as e:
                print(f"Error parsing pose: {e}")
                return None
        return [0, 0, 0, 0]

    def set_gripper(self, state):
        cmd = f"DigitalOutputs(1,{state})"
        self.send_command(self.client, cmd)

    # ============================================================
    # [NEW] Digital Output ของหัวดูด
    #   DO1 = Vacuum   (ดูด)
    #   DO2 = Pressure (เป่าลมปล่อยชิ้นงาน)
    # ใช้คำสั่ง DigitalOutputs(index,state) แบบเดียวกับ set_gripper เดิม
    # ============================================================
    DO_VACUUM = 1
    DO_PRESSURE = 2

    # firmware แต่ละรุ่นใช้ชื่อคำสั่งไม่เหมือนกัน (DigitalOutputs ตอบ -10000 = ไม่รู้จักคำสั่ง)
    # ลองตามลำดับ แล้วจำคำสั่งแรกที่หุ่นตอบรหัส 0 (สำเร็จ) ไว้ใช้ต่อ
    _DO_COMMANDS = ("DO", "DOExecute", "DigitalOutputs")
    _do_cmd = None

    def set_do(self, index, state):
        """สั่ง Digital Output: index = หมายเลข DO, state = 1 (ON) / 0 (OFF)"""
        state = 1 if state else 0
        if not self.use_real_robot:
            print(f"[MOCK DO] DO{index} = {state}")
            return ""
        names = [self._do_cmd] if self._do_cmd else list(self._DO_COMMANDS)
        res = ""
        for name in names:
            res = self.send_command(self.client, f"{name}({index},{state})")
            # [NEW] พิมพ์คำตอบของหุ่นทุกครั้ง เพื่อดูว่า DO ทำงานจริงหรือไม่
            print(f"[DO] ส่ง {name}({index},{state}) -> หุ่นตอบ: {res.strip()}")
            if res.strip().split(",")[0] == "0":
                if self._do_cmd is None:
                    self._do_cmd = name
                    print(f"[INFO] สั่ง Digital Output ด้วยคำสั่ง {name}(index,status)")
                return res
        print(f"[WARNING] สั่ง DO{index} = {state} ไม่สำเร็จ ตอบกลับ: {res.strip()}")
        return res

    def vacuum_on(self):
        """DO1 Vacuum ON"""
        return self.set_do(self.DO_VACUUM, 1)

    def vacuum_off(self):
        """DO1 Vacuum OFF"""
        return self.set_do(self.DO_VACUUM, 0)

    def pressure_on(self):
        """DO2 Pressure ON"""
        return self.set_do(self.DO_PRESSURE, 1)

    def pressure_off(self):
        """DO2 Pressure OFF"""
        return self.set_do(self.DO_PRESSURE, 0)

    def suction_pick(self):
        """ดูดชิ้นงาน: Pressure OFF -> Vacuum ON"""
        self.pressure_off()
        self.vacuum_on()

    def suction_release(self, blow_time=0.5):
        """ปล่อยชิ้นงาน: Vacuum OFF -> Pressure ON (เป่าลม blow_time วินาที) -> Pressure OFF"""
        self.vacuum_off()
        self.pressure_on()
        time.sleep(blow_time)
        self.pressure_off()

    def close(self):
        if self.use_real_robot:
            self.send_command(self.client, "DisableRobot()")
            self.client.close()