# utils/grouping.py

import pandas as pd
import difflib
from itertools import combinations
import networkx as nx
import os

# 유사도 계산 함수
def calculate_combined_similarity(ocr1, ocr2):
    total_similarity = difflib.SequenceMatcher(None, ocr1, ocr2).ratio()

    last_ocr1 = ocr1[-4:]
    last_ocr2 = ocr2[-4:]
    last_similarity = difflib.SequenceMatcher(None, last_ocr1, last_ocr2).ratio()

    fifth_ocr1 = ocr1[-5] if len(ocr1) > 4 else ''
    fifth_ocr2 = ocr2[-5] if len(ocr2) > 4 else ''
    fifth_similarity = 1 if fifth_ocr1 == fifth_ocr2 else 0

    front_ocr1 = ocr1[:-5]
    front_ocr2 = ocr2[:-5]
    front_similarity = difflib.SequenceMatcher(None, front_ocr1, front_ocr2).ratio()

    weight_total = 0.5
    weight_last = 0.3
    weight_fifth = 0.05
    weight_front = 0.15

    combined_similarity = (
        (total_similarity * weight_total) +
        (last_similarity * weight_last) +
        (fifth_similarity * weight_fifth) +
        (front_similarity * weight_front)
    )

    return combined_similarity

# 데이터 그룹화 함수
def process_file(file_path, output_folder):
    data = pd.read_csv(file_path)

    # OCR 값들로부터 유사한 번호판 그룹화
    vehicle_groups = []
    ocr_pairs = combinations(data['ocr'].unique(), 2)

    for ocr1, ocr2 in ocr_pairs:
        similarity = calculate_combined_similarity(ocr1, ocr2)
        if similarity > 0.7:  # 유사도 임계값 조정 가능
            vehicle_groups.append((ocr1, ocr2, similarity))

    # 그래프 생성하여 그룹화
    G = nx.Graph()

    # 유사한 OCR 쌍에 대한 엣지 추가
    for ocr1, ocr2, similarity in vehicle_groups:
        G.add_edge(ocr1, ocr2, weight=similarity)

    # 연결된 컴포넌트 찾기
    components = list(nx.connected_components(G))

    # 각 OCR에 그룹 번호 매핑
    ocr_to_group = {}
    for group_id, component in enumerate(components):
        for ocr in component:
            ocr_to_group[ocr] = group_id

    # 그룹 정보 추가
    data['group'] = data['ocr'].map(ocr_to_group)

    # 그룹별로 데이터 집계
    grouped_data = data.groupby('group').agg({
        'ocr': list,
        'accuracy': 'mean',
        'frame': ['min', 'max'],
        'TIME': ['min', 'max'],
        'direction': lambda x: x.mode().iloc[0] if not x.mode().empty else None
    }).reset_index()

    # 컬럼 이름 정리
    grouped_data.columns = [
        'group', 'ocr_list', 'accuracy', 'min_frame', 'max_frame', 'entry_time', 'exit_time', 'direction'
    ]

    # 결과 저장
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    grouped_file_path = os.path.join(output_folder, f'{base_name}_grouped.csv')
    grouped_data.to_csv(grouped_file_path, index=False)
    print(f"그룹화된 데이터 저장 완료: {grouped_file_path}")
