import cv2
import numpy as np
import pyautogui
import socket
import struct
import time
import threading

TCP_PORT = 8089
UDP_PORT = 8090  # Port used for automatic discovery


# --- Background Thread: Broadcast IP Address ---
def broadcast_presence():
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    print("📡 Automatic discovery beacon started...")
    while True:
        try:
            # Broadcast "STREAM_SERVER_HERE" to the entire local network
            udp_socket.sendto(b"STREAM_SERVER_HERE", ('255.255.255.255', UDP_PORT))
            time.sleep(1)  # Broadcast every 1 second
        except Exception:
            pass


# Start the discovery beacon in the background
threading.Thread(target=broadcast_presence, daemon=True).start()

# --- Main Thread: Handle Video Stream ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", TCP_PORT))
server_socket.listen(5)

print(f"🖥️  SERVER ACTIVE. Listening for auto-connecting clients on TCP port {TCP_PORT}...")

try:
    client_socket, client_address = server_socket.accept()
    print(f"🤝 Client found and connected automatically: {client_address}")

    while True:
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Scale down resolution to save bandwidth
        scale_percent = 0.7
        width = int(frame.shape[1] * scale_percent)
        height = int(frame.shape[0] * scale_percent)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Compress to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
        result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)

        if not result:
            continue

        data = encoded_frame.tobytes()
        size = len(data)

        # Send size header then image data
        client_socket.sendall(struct.pack(">L", size) + data)

except (ConnectionResetError, BrokenPipeError):
    print("\nClient disconnected.")
finally:
    server_socket.close()
    print("Server shut down safely.")
