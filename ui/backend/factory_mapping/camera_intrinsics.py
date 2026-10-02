"""ChArUco intrinsic calibration for the configured DFK camera."""
import math

import cv2
import numpy as np

from .calibration_data import parse_intrinsics


SQUARES = (24, 16)
SQUARE_LENGTH = 0.025
MARKER_LENGTH = 0.020
MIN_VIEWS = 8
MAX_VIEWS = 40


def board():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    if hasattr(aruco, 'CharucoBoard'):
        return aruco.CharucoBoard(SQUARES, SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    return aruco.CharucoBoard_create(*SQUARES, SQUARE_LENGTH, MARKER_LENGTH, dictionary)


def printable_board():
    target = board()
    image = target.generateImage((2400, 1600), marginSize=0) if hasattr(target, 'generateImage') else target.draw((2400, 1600), marginSize=0)
    success, encoded = cv2.imencode('.png', image)
    if not success: raise ValueError('Could not render calibration board')
    return encoded.tobytes()


def detect(image):
    target = board()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, 'CharucoDetector'):
        corners, ids, _, _ = cv2.aruco.CharucoDetector(target).detectBoard(gray)
    else:
        markers, marker_ids, _ = cv2.aruco.detectMarkers(gray, target.dictionary)
        if marker_ids is None: return None
        _, corners, ids = cv2.aruco.interpolateCornersCharuco(markers, marker_ids, gray, target, minMarkers=1)
    if ids is None or corners is None or len(ids) < 12: return None
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    height, width = gray.shape
    spread = np.ptp(corners, axis=0)
    if spread[0] * spread[1] < width * height * .02: return None
    return corners, ids


def add_view(samples, image, frame_id):
    if any(sample['frame_id'] == frame_id for sample in samples): raise ValueError('Wait for a new camera frame')
    if len(samples) >= MAX_VIEWS: raise ValueError('Maximum number of views reached; calibrate or reset')
    detected = detect(image)
    if detected is None: raise ValueError('Show more of the ChArUco board at a different angle')
    corners, ids = detected
    height, width = image.shape[:2]
    center = corners.mean(axis=0) / (width, height)
    area = np.prod(np.ptp(corners, axis=0) / (width, height))
    for sample in samples:
        if np.linalg.norm(center - sample['center']) < .035 and abs(area - sample['area']) < .035:
            raise ValueError('Move or rotate the board before capturing another view')
    samples.append(dict(frame_id=frame_id, corners=corners, ids=ids, size=(width, height), center=center, area=area))
    return dict(views=len(samples),corners=len(ids),required=MIN_VIEWS)


def calibrate(samples, camera):
    if len(samples) < MIN_VIEWS: raise ValueError(f'Capture at least {MIN_VIEWS} distinct board views')
    size = (camera['width'], camera['height'])
    if any(sample['size'] != size for sample in samples): raise ValueError('Camera resolution changed during calibration')
    object_corners = board().getChessboardCorners()
    objects = [np.asarray(object_corners[sample['ids']], dtype=np.float32).reshape(-1, 3) for sample in samples]
    images = [sample['corners'].reshape(-1, 2) for sample in samples]
    rms, matrix, distortion, *_ = cv2.calibrateCameraExtended(objects, images, size, None, None)
    if not math.isfinite(rms) or rms > 2 or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError('Calibration reprojection error is too high; capture clearer, varied views')
    coefficients = distortion.reshape(-1).tolist()
    if len(coefficients) != 5: raise ValueError('Unexpected distortion model from OpenCV')
    projection = np.column_stack((matrix, np.zeros(3)))
    def data(values):
        values = np.asarray(values)
        return dict(rows=values.shape[0],cols=values.shape[1],data=values.reshape(-1).tolist())
    result = dict(image_width=size[0],image_height=size[1],camera_name=camera['camera_name'],distortion_model='plumb_bob',
                  camera_matrix=data(matrix),distortion_coefficients=data(distortion.reshape(1, 5)),
                  rectification_matrix=data(np.eye(3)),projection_matrix=data(projection))
    parse_intrinsics(result, camera)
    return result, dict(rms=rms,views=len(samples),corners=sum(len(sample['ids']) for sample in samples))