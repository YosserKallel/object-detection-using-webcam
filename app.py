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

while True:
    success, frame = webcamera.read()
    if not success:
        continue

    results = model.predict(frame, conf=0.35, iou=0.5, imgsz=640, verbose=False)
    result = results[0]
    annotated_frame = frame.copy()
    frame_height, frame_width = annotated_frame.shape[:2]
    margin = 12
    stats_band_height = 48
    stats_band_width = 220
    tick_step = 120
    axis_color = (215, 215, 215)
    grid_color = (190, 190, 190)
    axis_tick_length = 7
    label_offset = 6

    grid_overlay = annotated_frame.copy()
    for x in range(tick_step, frame_width, tick_step):
        cv2.line(grid_overlay, (x, 0), (x, frame_height - 1), grid_color, 1, cv2.LINE_AA)
    for y in range(tick_step, frame_height, tick_step):
        cv2.line(grid_overlay, (0, y), (frame_width - 1, y), grid_color, 1, cv2.LINE_AA)
    cv2.addWeighted(grid_overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)

    axis_overlay = annotated_frame.copy()

    cv2.line(axis_overlay, (0, 0), (frame_width - 1, 0), axis_color, 1, cv2.LINE_AA)
    cv2.line(axis_overlay, (0, 0), (0, frame_height - 1), axis_color, 1, cv2.LINE_AA)

    for x in range(tick_step, frame_width, tick_step):
        cv2.line(axis_overlay, (x, 0), (x, axis_tick_length), axis_color, 1, cv2.LINE_AA)

    for y in range(tick_step, frame_height, tick_step):
        cv2.line(axis_overlay, (0, y), (axis_tick_length, y), axis_color, 1, cv2.LINE_AA)

    cv2.addWeighted(axis_overlay, 0.45, annotated_frame, 0.55, 0, annotated_frame)

    draw_translucent_rect(
        annotated_frame,
        (margin, margin),
        (margin + stats_band_width, margin + stats_band_height),
        (0, 0, 0),
        0.35,
    )
    cv2.putText(
        annotated_frame,
        f"Total: {len(result.boxes)}",
        (margin + 10, margin + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    draw_text_box(annotated_frame, "(0, 0)", (margin + 10, margin + stats_band_height + 8), font_scale=0.45)

    for x in range(tick_step, frame_width, tick_step):
        label_x = max(margin + axis_tick_length + label_offset, x - 18)
        draw_text_box(
            annotated_frame,
            str(x),
            (label_x, margin + stats_band_height + 6),
            font_scale=0.45,
            text_color=(230, 230, 230),
            alpha=0.45,
        )

    for y in range(tick_step, frame_height, tick_step):
        label_y = max(margin + axis_tick_length + label_offset, y - 10)
        draw_text_box(
            annotated_frame,
            str(y),
            (margin + axis_tick_length + label_offset, label_y),
            font_scale=0.45,
            text_color=(230, 230, 230),
            alpha=0.45,
        )

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = result.names[class_id]

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(annotated_frame, (center_x, center_y), 3, (255, 0, 0), -1)

        label_text = f"{class_name} {confidence:.2f}"
        label_width, label_height = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_top = y1 - label_height - 12
        if label_top < margin + stats_band_height + 12:
            label_top = y2 + 8
        draw_text_box(
            annotated_frame,
            label_text,
            (x1, label_top),
            font_scale=0.5,
        )

        coordinate_text = f"({center_x}, {center_y})"
        coordinate_top = center_y + 12
        if coordinate_top + 24 > frame_height - margin:
            coordinate_top = center_y - 36
        draw_text_box(
            annotated_frame,
            coordinate_text,
            (center_x + 10, coordinate_top),
            font_scale=0.45,
            text_color=(230, 230, 230),
            alpha=0.5,
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
