import asyncio
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from starlette.concurrency import run_in_threadpool

from color_analysis_mediapipe import analyze_personal_color
import uvicorn

app = FastAPI()


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
# PyCharm 실행 시
if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
