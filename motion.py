import cv2
import numpy as np
import concurrent.futures
import time
import os
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms


# Load the video
vid = cv2.VideoCapture("C:/Users/giaco/OneDrive/Desktop/sample.mp4")

# Verifica se la GPU è disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#torch
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load('C:/Users/giaco/OneDrive/Desktop/model_weights.pth', map_location=torch.device('cpu')))
# Sposta il modello sulla GPU
model = model.to(device)
model.eval()  # Imposta il modello in modalità di valutazione

# Create a directory to save detected areas
output_dir = "C:/Users/giaco/OneDrive/Desktop/detected_areas"
os.makedirs(output_dir, exist_ok=True)

# Read the first two frames for analysis
_, frame1 = vid.read()
_, frame2 = vid.read()

# Dictionary to track the centers of each identified contour over time
all_centers = {}
id_counter = 0  # Counter for assigning unique IDs to contours
frame_count = 0  # Count of processed frames

# Function to find the nearest contour
def find_nearest_contour(prev_centers, new_center, max_distance=50):
    nearest_id = None
    min_distance = max_distance
    for contour_id, prev_center in prev_centers.items():
        distance = np.sqrt((new_center[0] - prev_center[-1][0])**2 + (new_center[1] - prev_center[-1][1])**2)
        if distance < min_distance:
            nearest_id = contour_id
            min_distance = distance
    return nearest_id

# Definizione della trasformazione dell'immagine per adattarla a ResNet50
transform = transforms.Compose([
    transforms.ToPILImage(),  # Converti l'array OpenCV a immagine PIL
    transforms.ToTensor(),  # Converti in tensore
])

# Modifica della funzione per includere la predizione con PyTorch
def process_region(region, x_offset, y_offset, frame_index):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    region_data = []  # Lista per memorizzare i dati dei contorni in questa regione

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        # Ignora i piccoli movimenti
        if cv2.contourArea(contour) < 1300:
            continue

        # Calcola il centro del contorno in coordinate globali
        center_x, center_y = x + x_offset + w // 2, y + y_offset + h // 2
        new_center = (center_x, center_y)

        # Ritaglia l'area rilevata e preparala per l'input del modello
        detected_area = frame1[y + y_offset:y + y_offset + h, x + x_offset:x + x_offset + w]
        
        # Applica le trasformazioni per il modello
        input_tensor = transform(detected_area).unsqueeze(0)  # Aggiungi una dimensione batch
        input_tensor = input_tensor.to(device)  # Sposta l'input sulla GPU

        # Passa l'immagine nel modello per ottenere una predizione
        with torch.no_grad():  # Disabilita il calcolo del gradiente per velocizzare l'inferenza
            output = model(input_tensor)
            prediction = torch.sigmoid(output).item()  # Applica Sigmoid per ottenere una probabilità

        # Determina l'etichetta in base alla probabilità
        label = "Person" if prediction > 0.5 else "Not a person"

        # Salva i dati della regione
        region_data.append({
            "bbox": (x + x_offset, y + y_offset, w, h),
            "center": new_center,
            "label": label
        })

    return region_data

fps = 0
# Multithreading loop for video frames
while vid.isOpened():
    start_time = time.time()

    diff = cv2.absdiff(frame1, frame2)

    # Divide the frame into 4 regions
    height, width = diff.shape[:2]
    regions = [
        (diff[0:height // 2, 0:width // 2], 0, 0),
        (diff[0:height // 2, width // 2:width], width // 2, 0),
        (diff[height // 2:height, 0:width // 2], 0, height // 2),
        (diff[height // 2:height, width // 2:width], width // 2, height // 2),
    ]

    # Process each region in parallel
    new_centers = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_region, region, x_offset, y_offset, frame_count) for region, x_offset, y_offset in regions]
        for future in concurrent.futures.as_completed(futures):
            region_data = future.result()
            for item in region_data:
                # Get bounding box and center data
                bbox = item["bbox"]
                center = item["center"]
                label = item["label"]

                # Draw bounding box
                x, y, w, h = bbox
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw label
                cv2.putText(frame1, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Track contour center
                if frame_count > 0:
                    contour_id = find_nearest_contour(all_centers, center)
                    if contour_id is None:
                        contour_id = id_counter
                        id_counter += 1
                else:
                    contour_id = id_counter
                    id_counter += 1

                if contour_id not in all_centers:
                    all_centers[contour_id] = []
                all_centers[contour_id].append(center)

    # Draw paths for all tracked contours
    for contour_id, centers in all_centers.items():
        for i in range(len(centers) - 1):
            cv2.line(frame1, centers[i], centers[i + 1], (255, 0, 0), 2)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame1, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with bounding boxes and paths
    cv2.imshow("threshold", frame1)

    # Update frames for the next iteration
    frame1 = frame2
    ret, frame2 = vid.read()

    # Update frame count
    frame_count += 1

    # Exit loop on ESC key press
    if cv2.waitKey(40) == 27 or not ret:
        break

# Release video and close all windows
cv2.destroyAllWindows()
vid.release()
