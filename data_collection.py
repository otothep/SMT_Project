import cv2
import mediapipe as mp
import pandas as pd
import time

def collect_data():
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose()

    data = []

    def collect_samples(label):
        nonlocal data
        print(f"Positioniere deine rechte Hand und zeige auf den {label} Lautsprecher.")
        print("Die Datenerfassung beginnt in 5 Sekunden...")
        for i in range(5, 0, -1):
            print(i)
            time.sleep(1)

        for _ in range(100):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                row = [lm.x for lm in landmarks] + [lm.y for lm in landmarks] + [label]
                data.append(row)

                if len(data) % 10 == 0:
                    print(f"{len(data)} samples collected...")

    collect_samples("linken")  # Linker Lautsprecher
    collect_samples("rechten")  # Rechter Lautsprecher

    cap.release()
    df = pd.DataFrame(data)
    df.to_csv('pose_data.csv', index=False, header=False)

if __name__ == "__main__":
    collect_data()
