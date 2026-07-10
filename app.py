# pip install opencv-python

import cv2
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
print(model.names)
webcamera = cv2.VideoCapture(0)
# webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


def draw_translucent_rect(image, top_left, bottom_right, color, alpha):
    overlay = image.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def draw_rounded_rect(image, top_left, bottom_right, color, alpha, radius=10):
    overlay = image.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def draw_text_box(
    image,
    text,
    top_left,
    font_scale=0.5,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    alpha=0.5,
    padding_x=6,
    padding_y=4,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_width = text_width + (padding_x * 2)
    box_height = text_height + baseline + (padding_y * 2)

    x = max(0, min(top_left[0], image.shape[1] - box_width - 1))
    y = max(0, min(top_left[1], image.shape[0] - box_height - 1))

    draw_translucent_rect(image, (x, y), (x + box_width, y + box_height), bg_color, alpha)
    cv2.putText(
        image,
        text,
        (x + padding_x, y + text_height + padding_y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_text_only(image, text, position, font_scale=0.45, color=(255, 255, 255), thickness=1):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_text_with_shadow(image, text, position, font_scale=0.45, color=(255, 255, 255), thickness=1, shadow_offset=(1, 1)):
    shadow_position = (position[0] + shadow_offset[0], position[1] + shadow_offset[1])
    cv2.putText(
        image,
        text,
        shadow_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def place_total_badge(result, frame_width, frame_height, badge_width, badge_height, margin=20, safety=20):
    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append((int(x1), int(y1), int(x2), int(y2)))

    candidate_positions = [
        (margin, margin),
        (frame_width - badge_width - margin, margin),
        (margin, frame_height - badge_height - margin),
        (frame_width - badge_width - margin, frame_height - badge_height - margin),
    ]

    def is_clear(top_left):
        bx1, by1 = top_left
        bx2, by2 = bx1 + badge_width, by1 + badge_height
        bx1 -= safety
        by1 -= safety
        bx2 += safety
        by2 += safety
        for x1, y1, x2, y2 in boxes:
            if not (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2):
                return False
        return True

    for position in candidate_positions:
        if is_clear(position):
            return position

    y = margin
    while y + badge_height + margin < frame_height:
        position = (margin, y)
        if is_clear(position):
            return position
        y += 10

    return (margin, margin)

while True:
    success, frame = webcamera.read()
    if not success:
        continue

    results = model.predict(frame, conf=0.35, iou=0.5, imgsz=640, verbose=False)
    result = results[0]
    annotated_frame = frame.copy()
    frame_height, frame_width = annotated_frame.shape[:2]
    margin = 12
    stats_band_height = 42
    stats_band_width = 150
    tick_step = 120
    accent_color = (93, 202, 165)
    accent_dark = (44, 118, 97)
    axis_color = (245, 245, 245)
    grid_color = (245, 245, 245)
    axis_tick_length = 7
    label_offset = 6

    axis_overlay = annotated_frame.copy()

    cv2.line(axis_overlay, (0, 0), (frame_width - 1, 0), axis_color, 2, cv2.LINE_AA)
    cv2.line(axis_overlay, (0, 0), (0, frame_height - 1), axis_color, 2, cv2.LINE_AA)

    for x in range(tick_step, frame_width, tick_step):
        cv2.line(axis_overlay, (x, 0), (x, axis_tick_length), axis_color, 1, cv2.LINE_AA)

    for y in range(tick_step, frame_height, tick_step):
        cv2.line(axis_overlay, (0, y), (axis_tick_length, y), axis_color, 1, cv2.LINE_AA)

    cv2.addWeighted(axis_overlay, 0.60, annotated_frame, 0.40, 0, annotated_frame)

    total_text = f"Total: {len(result.boxes)}"
    total_text_width, total_text_height = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
    badge_width = total_text_width + 20
    badge_height = total_text_height + 14
    badge_x, badge_y = place_total_badge(result, frame_width, frame_height, badge_width, badge_height, margin=20, safety=20)

    draw_rounded_rect(
        annotated_frame,
        (badge_x, badge_y),
        (badge_x + badge_width, badge_y + badge_height),
        (20, 20, 20),
        0.55,
        radius=10,
    )
    draw_text_only(
        annotated_frame,
        total_text,
        (badge_x + 10, badge_y + total_text_height + 7),
        0.72,
        (255, 255, 255),
        2,
    )
    draw_text_only(
        annotated_frame,
        "(0, 0)",
        (margin + 2, margin + stats_band_height + 22),
        font_scale=0.4,
        color=(255, 255, 255),
        thickness=1,
    )

    for x in range(tick_step, frame_width, tick_step):
        draw_text_only(
            annotated_frame,
            str(x),
            (max(margin + axis_tick_length + label_offset, x - 14), 15),
            font_scale=0.5,
            color=(255, 255, 255),
            thickness=1,
        )

    for y in range(tick_step, frame_height, tick_step):
        draw_text_only(
            annotated_frame,
            str(y),
            (margin + axis_tick_length + label_offset, max(margin + 18, y - 6)),
            font_scale=0.5,
            color=(255, 255, 255),
            thickness=1,
        )

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = result.names[class_id]

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), accent_color, 2, cv2.LINE_AA)
        cv2.circle(annotated_frame, (center_x, center_y), 3, accent_color, -1)

        label_text = f"{class_name} {confidence:.2f}"
        label_width, label_height = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_box_top = max(margin + stats_band_height + 10, y1 - label_height - 14)
        label_box_left = x1
        draw_rounded_rect(
            annotated_frame,
            (label_box_left, label_box_top),
            (label_box_left + label_width + 14, label_box_top + label_height + 12),
            accent_color,
            0.24,
            radius=8,
        )
        draw_text_only(
            annotated_frame,
            label_text,
            (label_box_left + 7, label_box_top + label_height + 4),
            font_scale=0.5,
            color=accent_dark,
            thickness=1,
        )

        coordinate_text = f"({center_x}, {center_y})"
        coordinate_top = center_y + 12
        if coordinate_top + 24 > frame_height - margin:
            coordinate_top = center_y - 36
        draw_rounded_rect(
            annotated_frame,
            (center_x + 10, coordinate_top),
            (center_x + 10 + 92, coordinate_top + 24),
            (20, 20, 20),
            0.45,
            radius=7,
        )
        draw_text_only(
            annotated_frame,
            coordinate_text,
            (center_x + 16, coordinate_top + 16),
            font_scale=0.42,
            color=(255, 255, 255),
            thickness=1,
        )

    cv2.imshow("Live Camera", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()

# For Realsense camera
   # def initialize_realsense():
    #    import pyrealsense2 as rs
    #    pipeline = rs.pipeline()
     #   camera_aconfig = rs.config()
      #  camera_aconfig.enable_stream(rs.stream.depth, *config.DEPTH_CAMERA_RESOLUTION, rs.format.z16, config.DEPTH_CAMERA_FPS)
     #   camera_aconfig.enable_stream(rs.stream.color, *config.COLOR_CAMERA_RESOLUTION, rs.format.bgr8, COLOR_CAMERA_FPS)
     #   pipeline.start(camera_aconfig)
      #  return pipeline
# try:
#     # Try to initialize RealSense Camera
#     camera = initialize_realsense()
#     get_frame = get_frame_realsense
# except Exception as e:
#     print("RealSense camera not found, using default webcam.")
#     camera = initialize_webcam()
#     get_frame = get_frame_webcam

# Function to get frames from RealSense
# def get_frame_realsense(pipeline):
#     import pyrealsense2 as rs
#     frames = pipeline.wait_for_frames()
#     depth_frame = frames.get_depth_frame()
#     color_frame = frames.get_color_frame()
#     if not depth_frame or not color_frame:
#         return None, None
#     depth_image = np.asanyarray(depth_frame.get_data())
#     color_image = np.asanyarray(color_frame.get_data())
#     return depth_image, color_image

# # Function to get frame from webcam
# def get_frame_webcam(cap):
#     ret, frame = cap.read()
#     return None, frame if ret else None
