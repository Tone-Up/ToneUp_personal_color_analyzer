from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import os
import json

# 모델 및 프로세서 로드 (서버 시작 시 1회만)
MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)




def get_clip_embedding(image: Image.Image, text: str):
    """이미지 + 텍스트를 CLIP 임베딩으로 변환"""
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(inputs["pixel_values"])
        text_features = model.get_text_features(inputs["input_ids"])

    # L2 정규화하여 코사인 유사도 계산시 안정성 ↑
    image_emb = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    text_emb = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # 이미지 + 텍스트 임베딩을 합쳐서 최종 벡터 생성
    combined_emb = torch.cat([image_emb, text_emb], dim=-1)
    return combined_emb.cpu().numpy().tolist()


