import cv2
import numpy as np

def main():
    # เปิดกล้อง
    cap = cv2.VideoCapture(1)

    total_size_cm = 17.5
    num_squares = 8
    square_size_mm = (total_size_cm * 10.0) / num_squares  # 21.875 มม. ต่อช่อง
    pattern_size = (7, 7)  # จุดมุมภายใน 7x7

    print("--- โหมดคาลิเบตตารางหมากรุก ---")
    print("1. นำตารางหมากรุกมาวางให้กล้องเห็นชัดเจน")
    print("2. เมื่อเส้นตารางขึ้นสีเขียวครบถ้วน ให้กดปุ่ม 's' บนคีย์บอร์ดเพื่อบันทึกค่า")
    print("3. กดปุ่ม 'q' เพื่อออก")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถเปิดกล้องได้")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # วาดตาราง
            cv2.drawChessboardCorners(frame, pattern_size, corners_refined, found)

            cols, rows = pattern_size
            objp = np.zeros((rows * cols, 2), np.float32)
            for i in range(rows):
                for j in range(cols):
                    objp[i * cols + j] = [j * square_size_mm, i * square_size_mm]

            src_pts = corners_refined.reshape(-1, 2)
            dst_pts = objp
            h_matrix, _ = cv2.findHomography(src_pts, dst_pts)

            cv2.putText(frame, "Status: Detected! Press 's' to Save", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # เช็คถ้ากดปุ่ม 's' ให้บันทึกค่า Homography Matrix ลงไฟล์
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                np.save('homography_matrix.npy', h_matrix)
                print("✅ บันทึกค่าลงไฟล์ 'homography_matrix.npy' เรียบร้อยแล้ว! ปิดโปรแกรมนี้ได้เลย")
                break
        else:
            cv2.putText(frame, "Status: Searching for Chessboard...", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Calibration Window", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()