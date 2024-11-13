# utils/sami_processing.py

import re
from datetime import datetime, timedelta
import pandas as pd
import os

# SAMI 파일에서 시작 시간과 FPS 정보 추출
def extract_info_from_sami(sami_file_path):
    with open(sami_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 시작 시간 추출
    start_time_pattern = re.compile(r'<SYNC Start=0000><P Class=SUBTTL>(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})')
    start_time_match = start_time_pattern.search(content)
    if start_time_match:
        start_time = datetime.strptime(start_time_match.group(1), "%Y/%m/%d %H:%M:%S")
    else:
        raise ValueError(f"시작 시간을 찾을 수 없습니다: {sami_file_path}")

    # FPS 정보 추출
    fps_pattern = re.compile(r'FPS=(\d+)')
    fps_match = fps_pattern.search(content)
    if fps_match:
        fps = int(fps_match.group(1))
    else:
        raise ValueError(f"FPS 정보를 찾을 수 없습니다: {sami_file_path}")

    return start_time, fps

# 프레임 번호를 기준으로 시간을 계산하는 함수
def find_time_for_frame(frame_number, start_time, fps):
    elapsed_seconds = frame_number / fps
    actual_time = start_time + timedelta(seconds=elapsed_seconds)
    return actual_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

# SAMI 파일과 CSV 파일을 처리하여 TIME 열 추가
def process_files(sami_file_path, csv_file_path, output_folder):
    start_time, fps = extract_info_from_sami(sami_file_path)

    # CSV 파일 읽기
    df = pd.read_csv(csv_file_path)

    # TIME 열 추가
    df['TIME'] = df['frame'].apply(lambda x: find_time_for_frame(x, start_time, fps))

    # 결과 CSV 파일 저장
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_file_path = os.path.join(output_folder, f'{base_name}_time.csv')
    df.to_csv(output_file_path, index=False)
    print(f"시간 정보가 추가된 CSV 파일이 저장되었습니다: {output_file_path}")
