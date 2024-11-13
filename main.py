# main.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import torch
import easyocr
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from bson import ObjectId

from utils.video_processing import (
    process_video,
    load_model,
    get_model_paths,
    get_results_folder,
    get_processed_videos_folder,
    load_config
)
from utils.sami_processing import process_files as process_sami_files
from utils.csv_merging import merge_csv_files
from utils.grouping import process_file as process_grouping
from utils.encoding import convert_csv_encoding

# 환경 변수 로드
load_dotenv()

app = FastAPI()

# CORS 설정 (개발 단계에서 모든 도메인 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인만 허용하도록 변경하세요.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB 연결 설정
MONGO_URL = os.getenv('MONGO_URL')

if not MONGO_URL:
    raise ValueError("MONGO_URL 환경 변수가 설정되어 있지 않습니다.")

client = AsyncIOMotorClient(MONGO_URL)
db = client['caffein']  # 사용할 데이터베이스 이름으로 변경하세요.

# 설정 로드
_config = load_config()

# 필요한 디렉터리를 생성하는 함수
def create_directories():
    directories = [
        _config['uploaded_videos_folder'],
        _config['uploaded_sami_folder'],
        _config['results_folder'],
        _config['processed_videos_folder'],
        _config['sami_processed_results_folder'],
        _config['merged_csv_folder'],
        _config['grouped_results_folder'],
        _config['utf8_bom_results_folder'],
        _config['graphs_folder'],
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# 애플리케이션 시작 시 디렉터리 생성
@app.on_event("startup")
def startup_event():
    create_directories()

# Pydantic 모델 정의
class VideoMetadata(BaseModel):
    id: Optional[str] = Field(alias='_id')
    filename: str
    filepath: str
    upload_time: datetime
    status: str  # 'uploaded', 'processing', 'completed', 'error'
    result_path: Optional[str] = None
    csv_path: Optional[str] = None
    graph_path: Optional[str] = None
    is_short_video: Optional[bool] = None
    progress: Optional[int] = 0

# 모델 및 리더 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_paths = get_model_paths()
plate_model = load_model(model_paths['plate_model'], device)
color_model = load_model(model_paths['color_model'], device)
reader = easyocr.Reader(['ko'], gpu=torch.cuda.is_available())

# 결과 폴더 설정
results_folder = get_results_folder()
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

processed_videos_folder = get_processed_videos_folder()
if not os.path.exists(processed_videos_folder):
    os.makedirs(processed_videos_folder)

# 비디오 처리 엔드포인트
@app.post("/process_video/")
async def process_video_endpoint(video: UploadFile = File(...), sami: UploadFile = File(None)):
    video_folder = _config['uploaded_videos_folder']
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    video_path = os.path.join(video_folder, video.filename)

    # 업로드된 비디오 저장
    with open(video_path, "wb") as f:
        f.write(await video.read())

    # 메타데이터 생성
    video_metadata = {
        "filename": video.filename,
        "filepath": video_path,
        "upload_time": datetime.utcnow(),
        "status": "uploaded",
        "result_path": None,
        "csv_path": None,
        "graph_path": None,
        "is_short_video": None,
        "progress": 0,
    }

    # 데이터베이스에 메타데이터 삽입
    result = await db.videos.insert_one(video_metadata)
    video_id = str(result.inserted_id)

    # 영상 처리 시작
    await db.videos.update_one({"_id": ObjectId(video_id)}, {"$set": {"status": "processing"}})

    csv_filename = os.path.join(results_folder, f'{os.path.splitext(video.filename)[0]}_res.csv')
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps else 0
    cap.release()

    try:
        if duration < 300:  # 5분 = 300초
            # 영상 출력 포함하여 처리
            await process_short_video(video, sami, video_id)
            result_path = os.path.join(processed_videos_folder, f'processed_{video.filename}')
            await db.videos.update_one(
                {"_id": ObjectId(video_id)},
                {"$set": {
                    "status": "completed",
                    "result_path": result_path,
                    "csv_path": csv_filename,
                    "is_short_video": True,
                    "progress": 100,
                }}
            )
            return {"message": "영상 처리 완료 (짧은 영상)", "video_id": video_id}
        else:
            if sami is None:
                await db.videos.update_one({"_id": ObjectId(video_id)}, {"$set": {"status": "error"}})
                raise HTTPException(status_code=400, detail="긴 영상의 경우 SAMI 파일이 필요합니다.")
            await process_long_video(video, sami, video_id)
            result_path = csv_filename
            await db.videos.update_one(
                {"_id": ObjectId(video_id)},
                {"$set": {
                    "status": "completed",
                    "result_path": result_path,
                    "csv_path": csv_filename,
                    "is_short_video": False,
                    "progress": 100,
                }}
            )
            return {"message": "영상 처리 완료 (긴 영상)", "video_id": video_id}
    except Exception as e:
        await db.videos.update_one({"_id": ObjectId(video_id)}, {"$set": {"status": "error", "progress": 100}})
        raise HTTPException(status_code=500, detail=str(e))

async def process_short_video(video, sami, video_id):
    video_path = os.path.join(_config['uploaded_videos_folder'], video.filename)
    csv_filename = os.path.join(results_folder, f'{os.path.splitext(video.filename)[0]}_res.csv')
    # 영상 출력 포함하여 처리
    process_video(
        video_path,
        csv_filename,
        plate_model,
        color_model,
        reader,
        device,
        video_id=video_id,
        db=db,
        output_video=True
    )
    # SAMI 파일이 있으면 처리
    if sami is not None:
        sami_folder = _config['uploaded_sami_folder']
        if not os.path.exists(sami_folder):
            os.makedirs(sami_folder)
        sami_path = os.path.join(sami_folder, sami.filename)
        with open(sami_path, "wb") as f:
            f.write(await sami.read())
        output_folder = _config['sami_processed_results_folder']
        process_sami_files(sami_path, csv_filename, output_folder)

async def process_long_video(video, sami, video_id):
    video_path = os.path.join(_config['uploaded_videos_folder'], video.filename)
    csv_filename = os.path.join(results_folder, f'{os.path.splitext(video.filename)[0]}_res.csv')
    # 영상 출력 없이 처리
    process_video(
        video_path,
        csv_filename,
        plate_model,
        color_model,
        reader,
        device,
        video_id=video_id,
        db=db,
        output_video=False
    )
    # SAMI 파일 처리
    sami_folder = _config['uploaded_sami_folder']
    if not os.path.exists(sami_folder):
        os.makedirs(sami_folder)
    sami_path = os.path.join(sami_folder, sami.filename)
    with open(sami_path, "wb") as f:
        f.write(await sami.read())
    output_folder = _config['sami_processed_results_folder']
    process_sami_files(sami_path, csv_filename, output_folder)

# 처리 상태 조회 엔드포인트
@app.get("/video_status/{video_id}")
async def get_video_status(video_id: str):
    video = await db.videos.find_one({"_id": ObjectId(video_id)})
    if video:
        video_data = VideoMetadata(**video)
        return video_data
    else:
        raise HTTPException(status_code=404, detail="Video not found")

# 진행률 조회 엔드포인트
@app.get("/progress/{video_id}")
async def get_progress(video_id: str):
    video = await db.videos.find_one({"_id": ObjectId(video_id)})
    if video:
        progress = video.get("progress", 0)
        return {"progress": progress}
    else:
        raise HTTPException(status_code=404, detail="Video not found")

# 처리된 비디오 다운로드 엔드포인트
@app.get("/download_video/{video_id}")
async def download_video(video_id: str):
    video = await db.videos.find_one({"_id": ObjectId(video_id)})
    if video and video.get("result_path"):
        if video.get("is_short_video"):
            return FileResponse(path=video["result_path"], media_type='video/mp4', filename=f"processed_{video['filename']}")
        else:
            raise HTTPException(status_code=400, detail="긴 영상은 처리된 비디오가 없습니다.")
    else:
        raise HTTPException(status_code=404, detail="Processed video not found")

# CSV 파일 다운로드 엔드포인트
@app.get("/download_csv/{video_id}")
async def download_csv(video_id: str):
    video = await db.videos.find_one({"_id": ObjectId(video_id)})
    if video and video.get("csv_path"):
        return FileResponse(path=video["csv_path"], media_type='text/csv', filename=f"{os.path.basename(video['csv_path'])}")
    else:
        raise HTTPException(status_code=404, detail="CSV file not found")

# 그래프 이미지 다운로드 엔드포인트 (필요한 경우)
@app.get("/download_graph/{video_id}")
async def download_graph(video_id: str):
    video = await db.videos.find_one({"_id": ObjectId(video_id)})
    if video and video.get("graph_path"):
        return FileResponse(path=video["graph_path"], media_type='image/png', filename=f"{os.path.basename(video['graph_path'])}")
    else:
        raise HTTPException(status_code=404, detail="Graph image not found")

# CSV 병합 엔드포인트
@app.post("/merge_csv/")
async def merge_csv_endpoint():
    source_folder = _config['sami_processed_results_folder']
    output_folder = _config['merged_csv_folder']
    merge_csv_files(source_folder, output_folder)
    return {"message": "CSV 병합 완료", "output_folder": output_folder}

# 데이터 그룹화 엔드포인트
@app.post("/group_data/")
async def group_data_endpoint():
    source_folder = _config['merged_csv_folder']
    output_folder = _config['grouped_results_folder']
    for file in os.listdir(source_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(source_folder, file)
            process_grouping(file_path, output_folder)
    return {"message": "데이터 그룹화 완료", "output_folder": output_folder}

# 인코딩 변환 엔드포인트
@app.post("/convert_encoding/")
async def convert_encoding_endpoint():
    input_folder = _config['grouped_results_folder']
    output_folder = _config['utf8_bom_results_folder']
    convert_csv_encoding(input_folder, output_folder)
    return {"message": "인코딩 변환 완료", "output_folder": output_folder}

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
