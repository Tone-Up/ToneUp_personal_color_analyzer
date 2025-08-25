import asyncio
import json
import os
import time
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from PIL import Image
from starlette.responses import FileResponse

from color_analysis_mediapipe import analyze_personal_color
import uvicorn

from product_embedding import get_clip_embedding

app = FastAPI()

# 임베딩 저장 폴더
EMBEDDING_DIR = "embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# 서버에 저장된 이미지 경로
IMAGE_DIR = "product_images"  # 이미지가 저장된 폴더
os.makedirs(IMAGE_DIR, exist_ok=True)


class Product(BaseModel):
    id: int
    color: str = ""
    sex: str = ""
    type: str = ""

# @app.post("/analyze-color")
# async def analyze_color(user_id: int = Form(...),
#                         file: UploadFile = File(...)):
#     contents = await file.read()
#     result = analyze_personal_color(contents, user_id)
#     return {"userId": user_id, "personalColor": result}


@app.post("/analyze-color")
async def analyze_color(user_id: int = Form(...),
                        file: UploadFile = File(...)):
    semaphore = asyncio.Semaphore(12)  # 최대 5개 동시 요청 허용

    start = time.monotonic()
    async with semaphore:
        try:
            contents = await file.read()
            result = await run_in_threadpool(analyze_personal_color, contents, user_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            elapsed = time.monotonic() - start
            print(f"[analyze_color] 실행 시간: {elapsed:.3f}초")

        return {"userId": user_id, "personalColor": result}


@app.post("/embed")
async def embed_product(products: List[Product]):
    all_embeddings = []

    for product in products:

        # 서버에 저장된 이미지 경로
        image_path = os.path.join(IMAGE_DIR, f"{product.id}.jpg")
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"Image for product_id {product.id} not found"}

        img = Image.open(image_path).convert("RGB")

        text = make_clip_text(color_=product.color, gender=product.sex, type_=product.type)

        print(text+"  test")
        embedding = get_clip_embedding(img, text)
        all_embeddings.append({"product_id": product.id, "embedding": embedding})

        # save_path = os.path.join(EMBEDDING_DIR, f"{product.id}.json")
    output_file = os.path.join(EMBEDDING_DIR, "all_embeddings.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_embeddings, f, ensure_ascii=False)

    # 4. 파일을 직접 응답으로 반환 + 헤더로 status, total 추가
    headers = {
        "X-Status": "success",
        "X-Total": str(len(all_embeddings))
    }
    return FileResponse(
        path=output_file,
        filename="all_embeddings.json",
        media_type="application/json",
        headers=headers
    )


# PyCharm 실행 시
if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)


def make_clip_text(color_: str = "", gender: str = "", type_: str = ""):
    """
    상품 정보 기반으로 CLIP 텍스트 생성
    두 가지 형태 모두 반환
    """
    # # 1. 쉼표 방식
    # comma_text = ""
    # attrs = [color_, gender, type_]
    # attrs = [a for a in attrs if a]  # 빈 값 제거
    # if attrs:
    #     comma_text += ", " + ", ".join(attrs)

    # 2. 문장형 방식
    sentence_parts = []
    if color_:
        sentence_parts.append(f"{color_} ")
    if gender:
        sentence_parts.append(f"for {gender}")
    if type_:
        sentence_parts.append(type_)
    sentence_text = f"".join(sentence_parts)

    return sentence_text
