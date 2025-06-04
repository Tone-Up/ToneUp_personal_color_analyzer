import cv2
import numpy as np
import mediapipe as mp


def analyze_personal_color(image_bytes: bytes, user_id: int) -> str:
    # 이미지 bytes -> OpenCV 이미지 변환
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return "Error"

    img_origin = img.copy()
    img_eye_origin = img.copy()
    eye_mask = np.zeros_like(img)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return "Error"

        face_landmarks = results.multi_face_landmarks[0]

        # Mediapipe face mesh landmark index 참고:
        # 왼쪽 눈: [33, 133, 160, 159, 158, 157, 173, 246]
        # 오른쪽 눈: [263, 362, 387, 386, 385, 384, 398, 466]
        # 입: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

        h, w, _ = img.shape

        # 눈 좌표 추출 (convex hull 형태로 ellipse fitting 가능)
        left_eye_points = []
        right_eye_points = []
        mouth_points = []

        # 좌표 가져오기 함수
        def get_coords(idx_list):
            return np.array([(int(face_landmarks.landmark[i].x * w),
                              int(face_landmarks.landmark[i].y * h)) for i in idx_list], dtype=np.int32)

        left_eye_idx = [33, 133, 160, 159, 158, 157, 173, 246]
        right_eye_idx = [263, 362, 387, 386, 385, 384, 398, 466]
        mouth_idx = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

        left_eye = get_coords(left_eye_idx)
        right_eye = get_coords(right_eye_idx)
        mouth = get_coords(mouth_idx)

        # 각 영역 타원으로 마스크 생성
        left_eye_ellipse = cv2.fitEllipse(left_eye)
        right_eye_ellipse = cv2.fitEllipse(right_eye)
        mouth_ellipse = cv2.fitEllipse(mouth)

        cv2.ellipse(img, left_eye_ellipse, (0, 0, 0), -1)
        cv2.ellipse(img, right_eye_ellipse, (0, 0, 0), -1)
        cv2.ellipse(img, mouth_ellipse, (0, 0, 0), -1)

        eye_mask = cv2.ellipse(eye_mask, left_eye_ellipse, (255, 255, 255), -1)
        eye_mask = cv2.ellipse(eye_mask, right_eye_ellipse, (255, 255, 255), -1)

        # 얼굴 바운딩 박스 계산 (landmarks로 대략 계산)
        xs = [lm.x * w for lm in face_landmarks.landmark]
        ys = [lm.y * h for lm in face_landmarks.landmark]
        img_left, img_right = int(min(xs)), int(max(xs))
        img_top, img_bottom = int(min(ys)), int(max(ys))

    # 이하 원래 색상 분석 코드와 동일
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)

    ret_S, mat_S = cv2.threshold(S, -1, 255, cv2.THRESH_OTSU)
    ret_V, mat_V = cv2.threshold(V, -1, 255, cv2.THRESH_OTSU)
    S_th = ret_S
    V_th = ret_V

    S_val = 0.4
    V_val = 1.3

    low_lower = (0, int(S_th * S_val), int(V_th * V_val))
    low_upper = (30, 255, 255)
    high_lower = (150, int(S_th * S_val), int(V_th * V_val))
    high_upper = (179, 255, 255)

    img_mask_lower = cv2.inRange(img_hsv, low_lower, low_upper)
    img_mask_higher = cv2.inRange(img_hsv, high_lower, high_upper)
    img_mask = cv2.addWeighted(img_mask_lower, 1.0, img_mask_higher, 1.0, 0.0)

    img_merge = cv2.merge((img_mask, img_mask, img_mask))
    img_skin = cv2.bitwise_and(img, img_merge)
    img_eye = cv2.bitwise_and(img_eye_origin, eye_mask)
    img_eye_only = img_eye.copy()
    img_eye = cv2.cvtColor(img_eye, cv2.COLOR_BGR2HSV)
    He, Se, Ve = cv2.split(img_eye)
    ret_eye_S, eye_S = cv2.threshold(Se, -1, 255, cv2.THRESH_OTSU)
    eye_threshold_mask = cv2.inRange(img_eye, (0, ret_eye_S, 0), (255, 255, 255))
    img_eye_merge = cv2.merge((eye_threshold_mask, eye_threshold_mask, eye_threshold_mask))
    img_eye_converted = cv2.bitwise_and(img_eye, img_eye_merge)

    img_origin_roi = img_origin[img_top:img_bottom, img_left:img_right]
    img_roi = img_skin[img_top:img_bottom, img_left:img_right]
    img_eye_roi = img_eye_converted[img_top:img_bottom, img_left:img_right]

    r_sum = g_sum = b_sum = pixel_count = 0
    for y in range(img_bottom - img_top - 1):
        for x in range(img_right - img_left - 1):
            (b, g, r) = img_roi[y][x]
            if not (b == 0 and g == 0 and r == 0):
                r_sum += r
                g_sum += g
                b_sum += b
                pixel_count += 1
    if pixel_count == 0:
        return "Error"

    RGB_avg = (r_sum / pixel_count, g_sum / pixel_count, b_sum / pixel_count)
    RGB_Mat = np.full((1, 1, 3), (r_sum / pixel_count / 255, g_sum / pixel_count / 255, b_sum / pixel_count / 255),
                      np.float32)

    r_eye_sum = g_eye_sum = b_eye_sum = pixel_count = 0
    for y in range(img_bottom - img_top - 1):
        for x in range(img_right - img_left - 1):
            (b, g, r) = img_eye_roi[y][x]
            if not (b == 0 and g == 0 and r == 0):
                r_eye_sum += r
                g_eye_sum += g
                b_eye_sum += b
                pixel_count += 1
    if pixel_count == 0:
        return "Error"

    RGB_eye_avg = (r_eye_sum / pixel_count, g_eye_sum / pixel_count, b_eye_sum / pixel_count)
    RGB_eye_Mat = np.full((1, 1, 3),
                          (r_eye_sum / pixel_count / 255, g_eye_sum / pixel_count / 255, b_eye_sum / pixel_count / 255),
                          np.float32)

    LAB = cv2.cvtColor(RGB_Mat, cv2.COLOR_RGB2LAB)
    (l, a, b) = LAB[0][0]

    HSV = cv2.cvtColor(RGB_Mat, cv2.COLOR_RGB2HSV)
    HSV_eye = cv2.cvtColor(RGB_eye_Mat, cv2.COLOR_RGB2HSV)
    (h, s, v) = HSV[0][0]
    (he, se, ve) = HSV_eye[0][0]

    Tag = "Error"
    if a > b:
        if (se - s) > np.pi / 10:
            Tag = "WINTER" #"겨울 쿨톤"
        else:
            Tag =  "SUMMER"  #"여름 쿨톤"
    else:
        if (se - s) > np.pi / 10:
            Tag = "SPRING"       #"봄 웜톤"
        else:
            Tag = "ATUMN" #"가을 웜톤"

    return Tag
