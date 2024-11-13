import os
import re
import cv2
import csv
import gc
import json
import torch
import numpy as np
import easyocr
from ultralytics import YOLO
from tqdm import tqdm
from bson import ObjectId
import random  # 추가: 랜덤 색상 생성을 위해 임포트

# 번호판 유효성 검사를 위한 정규표현식
license_plate_pattern = re.compile(r'[0-9]{2,3}[가-힣]{1}[0-9]{4}')

# 설정 로드 함수
def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

_config = load_config()

def get_results_folder():
    return _config['results_folder']

def get_processed_videos_folder():
    return _config['processed_videos_folder']

def get_model_paths():
    return _config['model_paths']

# CSV 파일 초기화
def initialize_csv(csv_filename):
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['video', 'ID', 'color', 'ocr', 'accuracy', 'direction', 'frame'])

# CSV 파일에 데이터 저장
def save_to_csv(best_ocr_results, csv_filename):
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for data in best_ocr_results.values():
            writer.writerow([
                data['video'], data['ID'], data['color'],
                data['ocr'], data['accuracy'], data['direction'],
                data['frame']
            ])

# IoU 계산 함수
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou

# 객체 추적을 위한 함수
def assign_ids_to_boxes(boxes, state, best_ocr_results, csv_filename, max_frames_missing=20, iou_threshold=0.2):
    new_tracked_objects = {}
    unmatched_previous_objects = set(state['tracked_objects'].keys())

    for bbox in boxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        confidence = bbox.conf[0]
        cls = bbox.cls[0]
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        best_iou = 0
        best_id = None

        for obj_id, obj in state['tracked_objects'].items():
            existing_bbox = obj['bbox']
            iou = calculate_iou((x1, y1, x2, y2), existing_bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = obj_id

        if best_iou > iou_threshold:
            obj = state['tracked_objects'][best_id]
            new_tracked_objects[best_id] = {
                'bbox': (x1, y1, x2, y2),
                'center': center,
                'color': cls,
                'confidence': confidence,
                'best_ocr': obj['best_ocr'],
                'frames_missing': 0,
                'trajectory': obj['trajectory'] + [center],
                'direction': obj['direction']
            }
            unmatched_previous_objects.discard(best_id)
        else:
            new_tracked_objects[state['next_id']] = {
                'bbox': (x1, y1, x2, y2),
                'center': center,
                'color': cls,
                'confidence': confidence,
                'best_ocr': None,
                'frames_missing': 0,
                'trajectory': [center],
                'direction': None
            }
            state['next_id'] += 1

    # 오래 추적되지 않은 객체 정리
    for obj_id in unmatched_previous_objects:
        obj = state['tracked_objects'][obj_id]
        if obj['frames_missing'] >= max_frames_missing:
            if obj_id in best_ocr_results:
                save_to_csv({obj_id: best_ocr_results[obj_id]}, csv_filename)
                del best_ocr_results[obj_id]
                gc.collect()
        else:
            new_tracked_objects[obj_id] = {
                'bbox': obj['bbox'],
                'center': obj['center'],
                'color': obj['color'],
                'confidence': obj['confidence'],
                'best_ocr': obj['best_ocr'],
                'frames_missing': obj['frames_missing'] + 1,
                'trajectory': obj['trajectory'],
                'direction': obj['direction']
            }

    state['tracked_objects'] = new_tracked_objects
    return state

# 트래킹 상태 생성
def create_tracking_state():
    return {
        'tracked_objects': {},
        'next_id': 0,
        'id_colors': {}  # ID별 색상을 저장하기 위한 딕셔너리
    }

# 차량 ID별 색상 지정 함수
def id_to_color(obj_id, state):
    if 'id_colors' not in state:
        state['id_colors'] = {}
    if obj_id not in state['id_colors']:
        # 랜덤한 색상 생성
        color = tuple(random.randint(0, 255) for _ in range(3))
        state['id_colors'][obj_id] = color
    return state['id_colors'][obj_id]

# 입차/출차 방향 결정
def determine_direction(trajectory):
    if len(trajectory) >= 2:
        y_positions = [pos[1] for pos in trajectory]
        if y_positions[-1] - y_positions[0] > 50:
            return '입차'
        elif y_positions[0] - y_positions[-1] > 50:
            return '출차'
    return None

# 번호판 이미지 전처리
def preprocess_plate_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    morphed_img = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)
    morphed_img = cv2.equalizeHist(morphed_img)
    return morphed_img

# 모델 로드
def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    model = YOLO(model_path).to(device)
    return model

# 비디오 처리 함수
def process_video(video_path, csv_filename, plate_model, color_model, reader, device, video_id=None, db=None, batch_size=64, output_video=False):
    initialize_csv(csv_filename)
    best_ocr_results = {}
    state = create_tracking_state()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 영상 출력 설정
    if output_video:
        processed_videos_folder = get_processed_videos_folder()
        if not os.path.exists(processed_videos_folder):
            os.makedirs(processed_videos_folder)
        video_filename = os.path.basename(video_path)
        output_video_path = os.path.join(processed_videos_folder, f'processed_{video_filename}')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    else:
        out = None

    frame_count = 0
    batch_frames = []
    last_progress = 0  # 진행률 업데이트를 위한 변수

    with tqdm(total=total_frames, desc=f'Processing {os.path.basename(video_path)}') as pbar:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            batch_frames.append(img)

            # 배치 처리
            if len(batch_frames) == batch_size:
                process_batch(batch_frames, frame_count, plate_model, color_model, reader, device, best_ocr_results, state, csv_filename, output_video, out, video_path)
                frame_count += len(batch_frames)
                batch_frames = []
                pbar.update(batch_size)

                # 진행률 계산 및 업데이트
                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 1:
                    if db is not None and video_id:
                        db.videos.update_one(
                            {"_id": ObjectId(video_id)},
                            {"$set": {"progress": progress}}
                        )
                    last_progress = progress

        if len(batch_frames) > 0:
            # 남은 프레임 처리
            process_batch(batch_frames, frame_count, plate_model, color_model, reader, device, best_ocr_results, state, csv_filename, output_video, out, video_path)
            frame_count += len(batch_frames)
            pbar.update(len(batch_frames))

            # 진행률 계산 및 업데이트
            progress = int((frame_count / total_frames) * 100)
            if progress - last_progress >= 1:
                if db is not None and video_id:
                    db.videos.update_one(
                        {"_id": ObjectId(video_id)},
                        {"$set": {"progress": progress}}
                    )
                last_progress = progress

    cap.release()
    if output_video and out is not None:
        out.release()
    save_to_csv(best_ocr_results, csv_filename)

    # 처리 완료 후 진행률을 100%로 업데이트
    if db is not None and video_id:
        db.videos.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": {"progress": 100}}
        )
    return frame_count

def process_batch(batch_frames, frame_count, plate_model, color_model, reader, device, best_ocr_results, state, csv_filename, output_video, out, video_path):
    plate_results = plate_model.predict(source=batch_frames, imgsz=416, verbose=False)
    color_results = color_model.predict(source=batch_frames, imgsz=640, conf=0.3, verbose=False)

    # 각 프레임에 대한 처리
    for i, img in enumerate(batch_frames):
        current_frame_number = frame_count + i
        state = assign_ids_to_boxes(
            color_results[i].boxes, state, best_ocr_results, csv_filename, max_frames_missing=15
        )

        for obj_id, obj in state['tracked_objects'].items():
            x1, y1, x2, y2 = obj['bbox']
            center_x, center_y = obj['center']
            class_name = color_model.names[int(obj['color'])]
            confidence = obj['confidence']

            # 차량 ID별로 색상 지정
            color = id_to_color(obj_id, state)

            # 경계 상자 및 클래스 정보 그리기
            if output_video:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f'ID: {obj_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(img, f'Conf: {confidence:.2f}', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(img, f'Class: {class_name}', (x1, y1 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # 방향이 결정되면 표시
                if obj['direction'] is not None:
                    cv2.putText(img, f'Direction: {obj["direction"]}', (x1, y1 - 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if obj['direction'] is None:
                obj['direction'] = determine_direction(obj['trajectory'])
                if obj['direction'] is not None:
                    if obj_id in best_ocr_results:
                        best_ocr_results[obj_id]['direction'] = obj['direction']

        # 번호판 OCR 처리
        for bbox in plate_results[i].boxes:
            px1, py1, px2, py2 = map(int, bbox.xyxy[0])
            plate_confidence = bbox.conf[0]
            plate_color = (0, 0, 255)
            if output_video:
                cv2.rectangle(img, (px1, py1), (px2, py2), plate_color, 2)

            plate_center_x = (px1 + px2) // 2
            plate_center_y = (py1 + py2) // 2

            matched_obj_id = None
            for obj_id, obj in state['tracked_objects'].items():
                x1_obj, y1_obj, x2_obj, y2_obj = obj['bbox']
                if x1_obj < plate_center_x < x2_obj and y1_obj < plate_center_y < y2_obj:
                    matched_obj_id = obj_id
                    break

            if matched_obj_id is not None:
                plate_cropped_img = img[py1:py2, px1:px2]
                preprocessed_plate_img = preprocess_plate_image(plate_cropped_img)

                ocr_result = reader.readtext(preprocessed_plate_img)
                if ocr_result:
                    _, text, prob = ocr_result[0]
                    text = re.sub('[^가-힣0-9]', '', text)

                    if license_plate_pattern.fullmatch(text):
                        if (matched_obj_id not in best_ocr_results) or (prob > best_ocr_results[matched_obj_id]['accuracy']):
                            best_ocr_results[matched_obj_id] = {
                                'video': os.path.basename(video_path),
                                'ID': matched_obj_id,
                                'color': color_model.names[int(state['tracked_objects'][matched_obj_id]['color'])],
                                'ocr': text,
                                'accuracy': prob,
                                'direction': state['tracked_objects'][matched_obj_id]['direction'],
                                'frame': current_frame_number
                            }
                            state['tracked_objects'][matched_obj_id]['best_ocr'] = (text, prob)
                            if output_video:
                                color = id_to_color(matched_obj_id, state)
                                cv2.putText(img, f'OCR: {text}', (px1, py1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if output_video and out is not None:
            out.write(img)
