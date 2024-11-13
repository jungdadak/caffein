# utils/encoding.py

import os

# CSV 파일 인코딩 변환 함수
def convert_csv_encoding(input_folder, output_folder):
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.csv'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            # 기존 파일 읽기
            with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # UTF-8 with BOM으로 저장
            with open(output_file_path, 'w', encoding='utf-8-sig') as f:
                f.write(content)
            print(f"'{filename}' 파일을 UTF-8 with BOM으로 인코딩하여 '{output_folder}'에 저장했습니다.")
