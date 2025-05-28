from fastapi import FastAPI, UploadFile, File, Form
from color_analysis import analyze_personal_color
import uvicorn

app = FastAPI()


@app.post("/analyze-color")
async def analyze_color(user_id: int = Form(...),
                        file: UploadFile = File(...)):
    contents = await file.read()
    result = analyze_personal_color(contents, user_id)
    return {"id": user_id, "color": result}


# PyCharm 실행 시
if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
