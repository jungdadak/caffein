# utils/csv_merging.py

import os
import pandas as pd

# 폴더에서 CSV 파일을 날짜순으로 병합하는 함수
def merge_csv_files(source_folder, output_folder):
    # 결과를 저장할 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # CSV 파일 목록 가져오기 (파일 이름 사전순 정렬)
    csv_files = sorted([f for f in os.listdir(source_folder) if f.endswith('.csv')])

    # 날짜별로 CSV 파일 병합
    grouped_files = {}
    for csv_file in csv_files:
        # 파일 이름에서 날짜 부분 추출 (예: '20230101')
        base_name = csv_file.split('_')[0]

        if base_name not in grouped_files:
            grouped_files[base_name] = []
        grouped_files[base_name].append(csv_file)

    # 그룹별로 CSV 파일 병합
    for date, files in grouped_files.items():
        merged_df = pd.DataFrame()

        # 파일들을 순서대로 읽고 병합
        for file in files:
            file_path = os.path.join(source_folder, file)
            df = pd.read_csv(file_path)
            merged_df = pd.concat([merged_df, df], ignore_index=True)

        # 병합한 파일 저장
        output_file_path = os.path.join(output_folder, f'{date}_merged.csv')
        merged_df.to_csv(output_file_path, index=False)
        print(f"병합 완료: {output_file_path}")
