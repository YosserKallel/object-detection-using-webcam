# pip install opencv-python ultralytics
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import cv2
from ultralytics import YOLO
class ObjectPublisher(Node):

    def __init__(self):
        super().__init__('object_detector')
        self.publisher = self.create_publisher(
            PoseArray,
            '/detected_objects_xy',
            10
        )

    def publish_positions(self, points):
        """points: list of (x, y) tuples, one per detected object."""
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        for x, y in points:
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.0
            msg.poses.append(pose)
        self.publisher.publish(msg)
rclpy.init()

ros_node = ObjectPublisher()

model = YOLO('yolov8s.pt')

target_object = input("Enter object to follow: ").strip().lower()

print(f"Following only: {target_object}")

webcamera = cv2.VideoCapture(0)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
ACCENT = (229, 136, 30)          # bleu signal - seule couleur "action" de l'app (detections)
ACCENT_DARK = (50, 20, 0)        # texte fonce sur le badge bleu clair
BADGE_BG = (28, 28, 28)          # fond sombre des badges (total, coordonnees)
GRID_COLOR = (95, 95, 95)        # gris discret, en arriere-plan
AXIS_COLOR = (150, 120, 95)      # bleu-gris sourd - present mais ne domine pas l'image
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


def draw_axis_letter(image, letter, position, font_scale=0.55):
    """Lettre d'axe discrete, meme couleur sourde que les axes -> se fond
    dans la ligne plutot que d'attirer l'oeil comme un badge."""
    cv2.putText(image, letter, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, AXIS_COLOR, 1, cv2.LINE_AA)


def to_math_coords(px, py, origin_x, origin_y):
    """Convertit des coordonnees pixel (origine en haut a gauche, Y vers le bas)
    en coordonnees mathematiques (origine au centre, Y vers le haut)."""
    return px - origin_x, origin_y - py


def draw_repere(image, frame_width, frame_height, tick_step=100):
    """
    Grille + axes centres sur l'image, comme un repere mathematique :
    - l'origine (0,0) est au milieu du cadre
    - l'axe Y pointe vers le haut (valeurs positives au-dessus du centre)
    - l'axe X pointe vers la droite (valeurs positives a droite du centre)
    - chaque graduation affiche sa valeur "mathematique", pas sa position pixel
    """
    origin_x, origin_y = frame_width // 2, frame_height // 2

    grid_overlay = image.copy()
    for x in range(origin_x % tick_step, frame_width, tick_step):
        cv2.line(grid_overlay, (x, 0), (x, frame_height - 1), GRID_COLOR, 1, cv2.LINE_AA)
    for y in range(origin_y % tick_step, frame_height, tick_step):
        cv2.line(grid_overlay, (0, y), (frame_width - 1, y), GRID_COLOR, 1, cv2.LINE_AA)
    cv2.addWeighted(grid_overlay, 0.25, image, 0.75, 0, image)

    # --- axe X (horizontal, passe par le centre) ---
    cv2.line(image, (0, origin_y), (frame_width - 1, origin_y), AXIS_COLOR, 1, cv2.LINE_AA)

    # --- axe Y (vertical, passe par le centre) ---
    cv2.line(image, (origin_x, 0), (origin_x, frame_height - 1), AXIS_COLOR, 1, cv2.LINE_AA)

    # --- graduations X : valeur mathematique = x_pixel - origin_x ---
    for x in range(origin_x % tick_step, frame_width, tick_step):
        math_x = x - origin_x
        if math_x == 0:
            continue
        draw_label_chip(image, str(math_x), (x - 12, origin_y + 20), font_scale=0.4)

    # --- graduations Y : valeur mathematique = origin_y - y_pixel (inversee) ---
    for y in range(origin_y % tick_step, frame_height, tick_step):
        math_y = origin_y - y
        if math_y == 0:
            continue
        draw_label_chip(image, str(math_y), (origin_x + 8, y + 5), font_scale=0.4)

    draw_label_chip(image, "(0,0)", (origin_x + 8, origin_y + 20), font_scale=0.4)

    # --- lettres "X" et "Y" a l'extremite de chaque axe ---
    draw_axis_letter(image, "X", (frame_width - 22, origin_y - 10))
    draw_axis_letter(image, "Y", (origin_x + 10, 20))

    return origin_x, origin_y


def place_total_badge(result, frame_width, frame_height, badge_width, badge_height, margin=20, safety=20):
    boxes = [tuple(map(int, box.xyxy[0].tolist())) for box in result.boxes]
    top_offset = 16
    candidates = [
        (margin, margin + top_offset),
        (frame_width - badge_width - margin, margin + top_offset),
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
    return (margin, margin + top_offset)


while True:
    success, frame = webcamera.read()
    if not success:
        continue

    results = model.predict(frame, conf=0.35, iou=0.5, imgsz=640, verbose=False)
    result = results[0]
    # Keep only the selected object
    selected_boxes = []

    for box in result.boxes:
        class_name = result.names[int(box.cls.item())].lower()

        if class_name == target_object:
            selected_boxes.append(box)

    annotated = frame.copy()
    frame_height, frame_width = annotated.shape[:2]
    margin = 12

    # --- repere (grille + axes centres), toujours en dessous de tout le reste ---
    origin_x, origin_y = draw_repere(annotated, frame_width, frame_height, tick_step=100)

    # --- badge total ---
    total_text = f"Total: {len(selected_boxes)}"
    (tw, th), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    badge_w, badge_h = tw + 24, th + 18
    bx, by = place_total_badge(result, frame_width, frame_height, badge_w, badge_h, margin=20, safety=24)
    draw_rounded_rect(annotated, (bx, by), (bx + badge_w, by + badge_h), BADGE_BG, 0.6, radius=10)
    draw_text(annotated, total_text, (bx + 12, by + th + 9), 0.72, TEXT_WHITE, 2)

    # --- detections ---
    detected_points = []
    for box in selected_boxes:
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

        # coordonnees affichees en repere mathematique (centre, Y vers le haut)
        math_cx, math_cy = to_math_coords(cx, cy, origin_x, origin_y)
        detected_points.append((math_cx, math_cy))
        coord_text = f"({math_cx}, {math_cy})"
        coord_top = cy + 12 if cy + 36 < frame_height - margin else cy - 36
        draw_rounded_rect(annotated, (cx + 10, coord_top), (cx + 102, coord_top + 24),
                           BADGE_BG, 0.6, radius=6)
        draw_text(annotated, coord_text, (cx + 16, coord_top + 16), 0.42, TEXT_WHITE, 1)

    ros_node.publish_positions(detected_points)
    cv2.imshow("Live Camera", annotated)
    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()

ros_node.destroy_node()
rclpy.shutdown()
