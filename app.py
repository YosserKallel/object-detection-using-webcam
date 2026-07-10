# pip install opencv-python ultralytics

import cv2
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
webcamera = cv2.VideoCapture(0)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
ACCENT = (93, 202, 165)          # vert teal - couleur des detections
ACCENT_DARK = (24, 60, 50)       # texte fonce sur badge vert clair
BADGE_BG = (28, 28, 28)          # fond sombre des badges (total, coordonnees)
GRID_COLOR = (90, 90, 90)        # gris moyen, visible sur fond clair ET fonce
AXIS_COLOR = (0, 255, 255)       # jaune vif -> tres visible sur fond clair ET fonce
AXIS_OUTLINE = (0, 0, 0)         # contour noir pour renforcer le contraste
TEXT_WHITE = (255, 255, 255)


def draw_rounded_rect(image, top_left, bottom_right, color, alpha, radius=8):
    overlay = image.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def draw_text(image, text, position, font_scale=0.45, color=TEXT_WHITE, thickness=1):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_label_chip(image, text, position, font_scale=0.4, text_color=TEXT_WHITE,
                     bg_color=(20, 20, 20), alpha=0.55, pad=4):
    """Petit fond semi-transparent derriere le texte, pour rester lisible sur
    n'importe quel arriere-plan (mur clair ou sombre)."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x, y = position
    draw_rounded_rect(image, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad),
                       bg_color, alpha, radius=5)
    draw_text(image, text, (x, y), font_scale, text_color, 1)


def draw_axis_letter(image, letter, position, font_scale=0.6):
    """Lettre d'axe sans badge : contour noir epais puis remplissage jaune,
    pour se fondre visuellement dans la ligne d'axe plutot que flotter en boite."""
    cv2.putText(image, letter, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, AXIS_OUTLINE, 4, cv2.LINE_AA)
    cv2.putText(image, letter, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, AXIS_COLOR, 2, cv2.LINE_AA)


def draw_repere(image, frame_width, frame_height, tick_step=100):
    """
    Grille + axes vraiment visibles quel que soit le fond :
    - grille secondaire en gris moyen semi-transparent (visible sur fond clair et fonce)
    - 2 axes principaux (x=0, y=0) en jaune vif avec contour noir -> tres marques
    - chaque valeur de graduation a son propre petit fond, donc toujours lisible
    - lettres "X" et "Y" affichees a l'extremite de chaque axe
    """
    grid_overlay = image.copy()
    for x in range(tick_step, frame_width, tick_step):
        cv2.line(grid_overlay, (x, 0), (x, frame_height - 1), GRID_COLOR, 1, cv2.LINE_AA)
    for y in range(tick_step, frame_height, tick_step):
        cv2.line(grid_overlay, (0, y), (frame_width - 1, y), GRID_COLOR, 1, cv2.LINE_AA)
    cv2.addWeighted(grid_overlay, 0.35, image, 0.65, 0, image)

    # --- axe X (horizontal, y=0) : contour noir puis ligne jaune par-dessus ---
    cv2.line(image, (0, 0), (frame_width - 1, 0), AXIS_OUTLINE, 5, cv2.LINE_AA)
    cv2.line(image, (0, 0), (frame_width - 1, 0), AXIS_COLOR, 3, cv2.LINE_AA)

    # --- axe Y (vertical, x=0) : contour noir puis ligne jaune par-dessus ---
    cv2.line(image, (0, 0), (0, frame_height - 1), AXIS_OUTLINE, 5, cv2.LINE_AA)
    cv2.line(image, (0, 0), (0, frame_height - 1), AXIS_COLOR, 3, cv2.LINE_AA)

    for x in range(tick_step, frame_width, tick_step):
        draw_label_chip(image, str(x), (x - 12, 22), font_scale=0.4)

    for y in range(tick_step, frame_height, tick_step):
        draw_label_chip(image, str(y), (10, y + 5), font_scale=0.4)

    draw_label_chip(image, "(0,0)", (10, 22), font_scale=0.4)

    # --- lettres "X" et "Y" a l'extremite de chaque axe ---
    # Texte "flottant" directement sur la ligne (contour noir epais + jaune fin),
    # meme traitement visuel que les axes -> se lit comme une prolongation de
    # la ligne plutot que comme un badge separe.
    draw_axis_letter(image, "X", (frame_width - 22, 34))
    draw_axis_letter(image, "Y", (6, frame_height - 12))


def place_total_badge(result, frame_width, frame_height, badge_width, badge_height, margin=20, safety=20):
    boxes = [tuple(map(int, box.xyxy[0].tolist())) for box in result.boxes]
    candidates = [
        (margin, margin),
        (frame_width - badge_width - margin, margin),
        (margin, frame_height - badge_height - margin),
        (frame_width - badge_width - margin, frame_height - badge_height - margin),
    ]

    def is_clear(top_left):
        bx1, by1 = top_left
        bx2, by2 = bx1 + badge_width, by1 + badge_height
        bx1, by1, bx2, by2 = bx1 - safety, by1 - safety, bx2 + safety, by2 + safety
        return all(bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2 for x1, y1, x2, y2 in boxes)

    for pos in candidates:
        if is_clear(pos):
            return pos
    return (margin, margin)


while True:
    success, frame = webcamera.read()
    if not success:
        continue

    results = model.predict(frame, conf=0.35, iou=0.5, imgsz=640, verbose=False)
    result = results[0]
    annotated = frame.copy()
    frame_height, frame_width = annotated.shape[:2]
    margin = 12

    # --- repere (grille + axes), toujours en dessous de tout le reste ---
    draw_repere(annotated, frame_width, frame_height, tick_step=100)

    # --- badge total ---
    total_text = f"Total: {len(result.boxes)}"
    (tw, th), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    badge_w, badge_h = tw + 24, th + 18
    bx, by = place_total_badge(result, frame_width, frame_height, badge_w, badge_h, margin=20, safety=24)
    draw_rounded_rect(annotated, (bx, by), (bx + badge_w, by + badge_h), BADGE_BG, 0.6, radius=10)
    draw_text(annotated, total_text, (bx + 12, by + th + 9), 0.72, TEXT_WHITE, 2)

    # --- detections ---
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        class_name = result.names[int(box.cls.item())]
        confidence = float(box.conf.item())

        cv2.rectangle(annotated, (x1, y1), (x2, y2), ACCENT, 2, cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), 3, ACCENT, -1)

        label_text = f"{class_name} {confidence:.2f}"
        (lw, lh), base = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_top = max(margin + 30, y1 - lh - 14)
        draw_rounded_rect(annotated, (x1, label_top), (x1 + lw + 14, label_top + lh + 12),
                           ACCENT, 0.85, radius=6)
        draw_text(annotated, label_text, (x1 + 7, label_top + lh + 4), 0.5, ACCENT_DARK, 1)

        coord_text = f"({cx}, {cy})"
        coord_top = cy + 12 if cy + 36 < frame_height - margin else cy - 36
        draw_rounded_rect(annotated, (cx + 10, coord_top), (cx + 102, coord_top + 24),
                           BADGE_BG, 0.6, radius=6)
        draw_text(annotated, coord_text, (cx + 16, coord_top + 16), 0.42, TEXT_WHITE, 1)

    cv2.imshow("Live Camera", annotated)
    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()