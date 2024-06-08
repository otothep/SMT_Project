import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mido
from mido import Message
from mediapipe import solutions
import time

# Set the MIDI port name to match your IAC Driver
port_name = "IAC-Driver Bus 2"

# Improved neural network for classification
class ImprovedNN(nn.Module):
    def __init__(self):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(66, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)  # Changed to 3 for three classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Function to send MIDI note on message
def send_midi_note_on(note):
    with mido.open_output(port_name) as outport:
        msg = Message('note_on', note=note)
        outport.send(msg)

# Function to send MIDI note off message
def send_midi_note_off(note):
    with mido.open_output(port_name) as outport:
        msg = Message('note_off', note=note)
        outport.send(msg)

# Function to send MIDI control change (CC) message
def send_midi_cc(control, value):
    with mido.open_output(port_name) as outport:
        msg = Message('control_change', control=control, value=value)
        outport.send(msg)

# Function to collect pose data
def collect_data():
    pose = solutions.pose.Pose()
    cap = cv2.VideoCapture(0)
    data = []

    def collect_samples(label, prompt):
        print(prompt)
        print("Die Datenerfassung beginnt in 5 Sekunden...")
        for i in range(5, 0, -1):
            print(i)
            cv2.waitKey(1000)

        for _ in range(100):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                row = [lm.x for lm in landmarks] + [lm.y for lm in landmarks]
                row.append(label)  # Label for the class
                data.append(row)
                print(f"{len(data)} samples collected...")

    # Collect data for left speaker
    collect_samples(0, "Positioniere deine rechte Hand und zeige auf den linken Lautsprecher.")

    # Collect data for right speaker
    collect_samples(1, "Positioniere deine rechte Hand und zeige auf den rechten Lautsprecher.")

    # Collect data for no pointing
    collect_samples(2, "Bewege deine Hand, ohne auf die Lautsprecher zu zeigen.")

    cap.release()
    np.savetxt('pose_data.csv', np.array(data), delimiter=',')

# Function to train the neural network model
def train_model():
    data = np.loadtxt('pose_data.csv', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    model = ImprovedNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    for epoch in range(50):  # Increased number of epochs to 50
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'pose_model.pth')

# Function to test MIDI output for left and right speakers
def test_midi_output():
    print("Test: Playing note C on left speaker")
    send_midi_cc(10, 0)  # Pan to left
    send_midi_note_on(60)
    send_midi_note_off(60)
    time.sleep(1)

    print("Test: Playing note C on right speaker")
    send_midi_cc(10, 127)  # Pan to right
    send_midi_note_on(60)
    send_midi_note_off(60)
    time.sleep(1)

# Function to play notes with panning
def play_notes_with_panning():
    model = ImprovedNN()
    model.load_state_dict(torch.load('pose_model.pth'))
    model.eval()

    cap = cv2.VideoCapture(0)
    pose = solutions.pose.Pose()

    minor_9_chord = [72, 68, 65, 60, 57]  # C minor 9 (C, Eb, G, Bb, D)
    major_7_chord = [64, 60, 57, 53]      # C major 7 (C, E, G, B)

    minor_9_index = 0
    major_7_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            data = [lm.x for lm in landmarks] + [lm.y for lm in landmarks]
            data = torch.tensor([data], dtype=torch.float32)
            output = model(data)
            _, predicted = torch.max(output, 1)

            if predicted.item() == 0:
                print("Playing note on left speaker")
                send_midi_cc(10, 0)  # Pan to left
                note = minor_9_chord[minor_9_index]
                send_midi_note_on(note)
                time.sleep(0.5)
                send_midi_note_off(note)
                minor_9_index = (minor_9_index + 1) % len(minor_9_chord)
            elif predicted.item() == 1:
                print("Playing note on right speaker")
                send_midi_cc(10, 127)  # Pan to right
                note = major_7_chord[major_7_index]
                send_midi_note_on(note)
                time.sleep(0.5)
                send_midi_note_off(note)
                major_7_index = (major_7_index + 1) % len(major_7_chord)

        cv2.imshow("Pose Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
    train_model()
    test_midi_output()
    play_notes_with_panning()
