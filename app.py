import os
import re
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# 페이지 설정
st.set_page_config(page_title="이미지 기반 PDP 자동화", layout="centered")
st.markdown("""
<style>
  .main .block-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
  }
</style>
""", unsafe_allow_html=True)

# 모델 로드
@st.cache_resource
def load_blip():
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    mdl = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return proc, mdl

@st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(['en'], gpu=False)

processor, blip_model = load_blip()
reader = load_easyocr_reader()

# UI: 이미지 업로드
st.title("이미지 기반 PDP 생성 자동화 솔루션")
uploaded = st.file_uploader("이미지를 업로드하세요", type=["png","jpg","jpeg"])
if not uploaded:
    st.info("이미지를 선택해주세요")
    st.stop()

img = Image.open(uploaded).convert("RGB")
annotated = img.copy()
draw = ImageDraw.Draw(annotated)
font = ImageFont.load_default()
draw.text((10, 10), "1", fill="red", font=font)
st.image(annotated, use_container_width=True)

# EasyOCR 텍스트 추출
def extract_text_via_easyocr(pil_img: Image.Image) -> str:
    arr = np.array(pil_img)
    lines = reader.readtext(arr, detail=0, paragraph=True)
    return "\n".join(lines)

# BLIP 캡션 생성
def generate_blip_caption(pil_img):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    blip_model.to(device)
    out = blip_model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)

# Alt 후보 생성
def make_alt_candidates(pil_img):
    base = generate_blip_caption(pil_img)
    cands = [base, base + " in a modern environment", "LG product - " + base]
    # 고유하게 최대 3개
    unique = []
    for c in cands:
        if c not in unique:
            unique.append(c)
        if len(unique) == 3:
            break
    return unique

# 분류/추천 정의
COMPONENT_DEFS = {
    "ST0001": {"name": "Hero banner", "has_image": True},
    "ST0002": {"name": "Tab Anchor", "has_image": False},
    "ST0003": {"name": "Title", "has_image": False},
    "ST0013": {"name": "Side Image", "has_image": True},
    "ST0014": {"name": "Layered Text", "has_image": True},
}

CTA_KEYWORDS = [
    "learn more","shop now","buy now","see more","read more",
    "click here","get started","try now","explore","discover","buy it now"
]

def classify_elements(lines):
    extracted, cleaned = [], []
    for L in lines:
        line = L
        for kw in CTA_KEYWORDS:
            if kw in line.lower():
                extracted.append(kw)
                line = re.sub(rf"\b{re.escape(kw)}\b", "", line, flags=re.IGNORECASE).strip()
        if line:
            cleaned.append(line)
    disclaimers = [l for l in cleaned if l.startswith("*")]
    body = [l for l in cleaned if not l.startswith("*")]
    ey = body.pop(0) if body and body[0].isupper() else ""
    hl = body.pop(0) if body else ""
    bc = "\n".join(body)
    return {"Eyebrow": ey, "Headline": hl, "Bodycopy": bc,
            "Disclaimer": "\n".join(disclaimers),
            "CTA": (extracted[0] if extracted else "")}

def recommend_components(classified, has_image=True):
    return [cid for cid, comp in COMPONENT_DEFS.items() if comp["has_image"] == has_image]

# Alt Text 생성
if st.button("🖼️ Alt Text 생성"):
    st.session_state["candidates"] = make_alt_candidates(img)
if "candidates" in st.session_state:
    choice = st.radio("Alt Text 후보 선택:", st.session_state["candidates"], key="alt_choice")
    st.subheader("🖼️ Selected Alt Text")
    st.code(choice)

# OCR 실행 (EasyOCR)
if st.button("🚀 OCR 실행 (EasyOCR)"):
    txt = extract_text_via_easyocr(img)
    lines = txt.split("\n")
    st.session_state["ocr_done"] = True
    st.session_state["ocr_text"] = txt
    st.session_state["classified"] = classify_elements(lines)
    st.session_state["recs"] = recommend_components(st.session_state["classified"])

# 결과 렌더링
if st.session_state.get("ocr_done"):
    st.subheader("📋 OCR 결과")
    st.text_area("", st.session_state["ocr_text"], height=200)
    st.subheader("📑 텍스트 분류 결과")
    for k, v in st.session_state["classified"].items():
        st.markdown(f"**{k}:** {v or '—'}")
    with st.expander("🧩 컴포넌트 추천"):
        recs = st.session_state["recs"]
        if recs:
            sel = st.selectbox("추천된 컴포넌트 중 하나를 선택하세요:", recs,
                               format_func=lambda cid: f"{cid} – {COMPONENT_DEFS[cid]['name']}")
            st.markdown(f"**Selected Component:** {sel} – {COMPONENT_DEFS[sel]['name']}")
            st.session_state["selected_component"] = sel
        else:
            st.write("추천할 컴포넌트가 없습니다.")
    df = pd.DataFrame.from_dict(st.session_state["classified"], orient='index', columns=['Content'])
    csv = df.to_csv(encoding='utf-8-sig')
    st.download_button("💾 엑셀 파일 다운로드", data=csv, file_name="ocr_result.csv", mime="text/csv")

# 전송 기능
if st.session_state.get("ocr_done") and st.button("📤 PDP생성하기 (WCM으로 전송하기)"):
    st.caption("※ 해당 기능은 기획 단계의 구현이며 실제 적용되어 있지 않습니다.")
