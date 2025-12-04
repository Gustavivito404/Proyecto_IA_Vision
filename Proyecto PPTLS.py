import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import random

# ==========================
# PANTALLAS DE TEXTO EN VEZ DE IMAGENES
# ==========================
def create_text_image(text, width=640, height=480, color=(255, 255, 255)):
    """
    Crea una imagen en negro con texto centrado aproximadamente.
    Soporta saltos de línea usando '\n'.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # fondo gris oscuro

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2

    # Dividir el texto en líneas
    lines = text.split("\n")
    line_height = 30  # separación vertical entre líneas

    # Calcular altura total del bloque de texto
    total_text_height = line_height * len(lines)
    # Primera línea un poco arriba del centro
    y0 = height // 2 - total_text_height // 2

    for i, line in enumerate(lines):
        (text_w, text_h), _ = cv2.getTextSize(line, font, scale, thickness)
        x = max(10, (width - text_w) // 2)
        y = y0 + i * line_height
        cv2.putText(img, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    cv2.putText(
        img,
        "ESC o Q para salir",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return img

def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid


def fingers_up_down(hand_results, thumb_points, palm_points, fingertips_points, finger_base_points):
    fingers = None
    coordinates_thumb = []
    coordinates_palm = []
    coordinates_ft = []
    coordinates_fb = []
    for hand_landmarks in hand_results.multi_hand_landmarks:
        for index in thumb_points:
            x = int(hand_landmarks.landmark[index].x * width)
            y = int(hand_landmarks.landmark[index].y * height)
            coordinates_thumb.append([x, y])

        for index in palm_points:
            x = int(hand_landmarks.landmark[index].x * width)
            y = int(hand_landmarks.landmark[index].y * height)
            coordinates_palm.append([x, y])

        for index in fingertips_points:
            x = int(hand_landmarks.landmark[index].x * width)
            y = int(hand_landmarks.landmark[index].y * height)
            coordinates_ft.append([x, y])

        for index in finger_base_points:
            x = int(hand_landmarks.landmark[index].x * width)
            y = int(hand_landmarks.landmark[index].y * height)
            coordinates_fb.append([x, y])

        # -------------------------
        # Pulgar
        # -------------------------
        p1 = np.array(coordinates_thumb[0])
        p2 = np.array(coordinates_thumb[1])
        p3 = np.array(coordinates_thumb[2])

        l1 = np.linalg.norm(p2 - p3)
        l2 = np.linalg.norm(p1 - p3)
        l3 = np.linalg.norm(p1 - p2)

        # Calcular el ángulo
        to_angle = (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)
        if int(to_angle) == -1:
            angle = 180
        else:
            angle = degrees(acos(to_angle))
        thumb_finger = np.array(False)
        if angle > 150:
            thumb_finger = np.array(True)

        # -------------------------
        # Índice, medio, anular y meñique
        # -------------------------
        nx, ny = palm_centroid(coordinates_palm)
        cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)
        coordinates_centroid = np.array([nx, ny])
        coordinates_ft_arr = np.array(coordinates_ft)
        coordinates_fb_arr = np.array(coordinates_fb)

        # Distancia entre la punta del pulgar (landmark 4) y la punta del índice (landmark 8)
        # coordinates_thumb[2] corresponde al punto 4 (pulgar)
        # coordinates_ft_arr[0] corresponde al punto 8 (índice)
        global thumb_index_dist
        thumb_tip  = np.array(coordinates_thumb[2])
        index_tip  = np.array(coordinates_ft_arr[0])
        thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)

        # Distancias
        d_centrid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft_arr, axis=1)
        d_centrid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb_arr, axis=1)
        dif = d_centrid_ft - d_centrid_fb
        fingers = dif > 0
        fingers = np.append(thumb_finger, fingers)

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    return fingers


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Pulgar
thumb_points = [1, 2, 4]

# Índice, medio, anular y meñique
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]

# ==========================
# FINGERS COMBINATIONS
# ==========================
TO_ACTIVATE = np.array([True, False, False, False, False])

# Piedra, papel, tijeras
PIEDRA = np.array([False, False, False, False, False])
PAPEL = np.array([True, True, True, True, True])
TIJERAS = np.array([False, True, True, False, False])

# Lagarto: índice + meñique arriba
LAGARTO = np.array([False, True, False, False, True])

# Spock: mano abierta sin pulgar (indice, medio, anular y meñique arriba)
SPOCK = np.array([False, True, True, True, True])

# REGLAS PIEDRA, PAPEL, TIJERAS, LAGARTO, SPOCK (0..4)
WIN_GAME = [
    "21",  # Tijeras corta Papel
    "10",  # Papel tapa Piedra
    "03",  # Piedra aplasta Lagarto
    "34",  # Lagarto envenena Spock
    "42",  # Spock rompe Tijeras
    "23",  # Tijeras decapita Lagarto
    "31",  # Lagarto come Papel
    "14",  # Papel refuta Spock
    "40",  # Spock vaporiza Piedra
    "02",  # Piedra rompe Tijeras
]

# Mapear número -> nombre de jugada
CHOICES = {
    0: "Piedra",
    1: "Papel",
    2: "Tijeras",
    3: "Lagarto",
    4: "Spock",
}

pc_option = False  # Si la pc ha escogido o no
detect_hand = True

THRESHOLD = 10
THRESHOLD_RESTART = 50

thumb_index_dist = None  # distancia entre punta de pulgar (4) e índice (8)

count_like = 0
count_piedra = 0
count_papel = 0
count_tijeras = 0
count_lagarto = 0
count_spock = 0
count_restart = 0

# ==========================
# "IMAGENES" AHORA SON PANTALLAS DE TEXTO
# ==========================
IMG_H = 480
IMG_W = 640

image1 = create_text_image(
    "Pulgar arriba para iniciar la partida",
    IMG_W,
    IMG_H,
    color=(255, 255, 255),
)
image2 = create_text_image(
    "PC lista. \nTu turno: Piedra/Papel/Tijeras/Lagarto/Spock",
    IMG_W,
    IMG_H,
    color=(255, 255, 255),
)

# Image to concat
imAux = image1

player = None

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            fingers = fingers_up_down(
                results,
                thumb_points,
                palm_points,
                fingertips_points,
                finger_base_points,
            )
            # print("fingers:", fingers)

            if detect_hand == True:
                # Pulgar arriba para que la PC elija
                if (not False in (fingers == TO_ACTIVATE)) and pc_option == False:
                    if count_like >= THRESHOLD:
                        pc = random.randint(0, 4)  # ahora 0..4
                        print("pc:", pc, CHOICES[pc])
                        pc_option = True
                        imAux = image2
                    count_like += 1

                if pc_option == True:
                    if not False in (fingers == PIEDRA):
                        if count_piedra >= THRESHOLD:
                            player = 0
                        count_piedra += 1

                    elif not False in (fingers == PAPEL):
                        # Umbral para considerar "boca cerrada" (ajustable)
                        LAGARTO_THRESHOLD = width * 0.10  # 10% del ancho de la imagen aprox.

                        if thumb_index_dist is not None and thumb_index_dist < LAGARTO_THRESHOLD:
                         # Lagarto tipo serie: papel pero con pulgar e índice muy juntitos
                            if count_lagarto >= THRESHOLD:
                                player = 3   # 3 = Lagarto
                            count_lagarto += 1
                        else:
                            # Papel normal
                            if count_papel >= THRESHOLD:
                                player = 1   # 1 = Papel
                            count_papel += 1

                    elif not False in (fingers == TIJERAS):
                        if count_tijeras >= THRESHOLD:
                            player = 2
                        count_tijeras += 1

                    elif not False in (fingers == SPOCK):
                        if count_spock >= THRESHOLD:
                            player = 4
                        count_spock += 1

        # ---- EVALUACIÓN DEL JUEGO (fuera del if de la mano) ----
        if player is not None:
            detect_hand = False
            if pc == player:
                texto_empate = f"Empate. Tu: {CHOICES[player]}, PC: {CHOICES[pc]}"
                imAux = create_text_image(texto_empate, IMG_W, IMG_H, color=(0, 255, 255))
            else:
                if (str(player) + str(pc)) in WIN_GAME:
                    # Ganaste: podemos mostrar qué sacó cada uno también
                    texto_ganaste = f"Ganaste. Tu: {CHOICES[player]}, PC: {CHOICES[pc]}"
                    imAux = create_text_image(texto_ganaste, IMG_W, IMG_H, color=(0, 255, 0))
                else:
                    texto_perdiste = f"Perdiste. Tu: {CHOICES[player]}, PC: {CHOICES[pc]}"
                    imAux = create_text_image(texto_perdiste, IMG_W, IMG_H, color=(0, 0, 255))

            count_restart += 1
            if count_restart > THRESHOLD_RESTART:
                pc_option = False
                detect_hand = True
                player = None

                count_like = 0
                count_piedra = 0
                count_papel = 0
                count_tijeras = 0
                count_lagarto = 0
                count_spock = 0
                count_restart = 0
                imAux = image1

        n_image = cv2.hconcat([imAux, frame])
        cv2.imshow("n_image", n_image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
