from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import base64
import time
import os
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

# ── Configuração ──────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
REJECT_THRESHOLD = float(os.getenv("REJECT_THRESHOLD", "0.25"))  # conf mínima

app = FastAPI(
    title="PCB Defect Detection API",
    description="API de Inspeção de Qualidade em PCBs usando YOLOv8",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Carregar modelo ────────────────────────────────────────────────────────────
print(f"[INFO] Carregando modelo: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    CLASS_NAMES = model.names  # dict {0: 'damaged', 1: 'lack_of_part', ...}
    print(f"[INFO] Modelo carregado. Classes: {CLASS_NAMES}")
except Exception as e:
    print(f"[ERRO] Falha ao carregar modelo: {e}")
    model = None
    CLASS_NAMES = {
        0: "damaged", 1: "lack_of_part", 2: "miss_welding",
        3: "redundant", 4: "Short_circuit", 5: "slug", 6: "spillover"
    }

# ── Histórico em memória ───────────────────────────────────────────────────────
inspection_history = []
session_stats = defaultdict(int)
total_inspected = 0
total_defective = 0

# ── Helpers ───────────────────────────────────────────────────────────────────
DEFECT_COLORS = {
    "damaged":       (0,   0,   255),   # vermelho
    "lack_of_part":  (0,   165, 255),   # laranja
    "miss_welding":  (0,   255, 255),   # amarelo
    "redundant":     (128, 0,   255),   # roxo
    "Short_circuit": (255, 0,   0  ),   # azul
    "slug":          (0,   128, 0  ),   # verde
    "spillover":     (255, 0,   128),   # rosa
}

def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Desenha bounding boxes e labels na imagem."""
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class"]
        conf = det["confidence"]
        color = DEFECT_COLORS.get(label, (200, 200, 200))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img

def compute_business_rules(detections: list, img_area: int) -> dict:
    """Regras de negócio: severidade, área defeituosa, recomendação."""
    if not detections:
        return {
            "status": "aprovado",
            "severity": "nenhum",
            "defect_area_pct": 0.0,
            "recommendation": "Peça aprovada. Nenhum defeito detectado.",
            "alert": False
        }

    # Área total das detecções
    total_defect_area = 0
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        total_defect_area += (x2 - x1) * (y2 - y1)

    defect_pct = min((total_defect_area / img_area) * 100, 100.0)

    # Defeitos críticos
    critical = {"Short_circuit", "miss_welding", "lack_of_part"}
    has_critical = any(d["class"] in critical for d in detections)
    num_defects = len(detections)

    if has_critical or defect_pct > 15 or num_defects >= 5:
        severity = "crítico"
        status = "reprovado"
        recommendation = "⛔ Peça REPROVADA. Defeito crítico detectado. Remover da esteira imediatamente."
        alert = True
    elif defect_pct > 5 or num_defects >= 2:
        severity = "moderado"
        status = "reprovado"
        recommendation = "⚠️ Peça REPROVADA. Múltiplos defeitos. Encaminhar para inspeção manual."
        alert = True
    else:
        severity = "leve"
        status = "reprovado"
        recommendation = "🔍 Peça REPROVADA. Defeito leve detectado. Monitorar lote."
        alert = False

    return {
        "status": status,
        "severity": severity,
        "defect_area_pct": round(defect_pct, 2),
        "recommendation": recommendation,
        "alert": alert
    }

def encode_image(img: np.ndarray) -> str:
    """Converte imagem numpy para base64 JPEG."""
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "PCB Defect Detection API", "status": "online", "version": "1.0.0"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "classes": CLASS_NAMES,
        "total_inspected": total_inspected,
        "total_defective": total_defective
    }

@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...), conf: float = REJECT_THRESHOLD):
    """
    Recebe uma imagem, roda inferência YOLO e retorna:
    - detecções com bounding boxes
    - imagem anotada em base64
    - métricas de negócio (severidade, área defeituosa, recomendação)
    """
    global total_inspected, total_defective

    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    # Ler imagem
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")

    h, w = img.shape[:2]
    img_area = h * w

    # Inferência
    t0 = time.time()
    results = model.predict(img, conf=conf, verbose=False)[0]
    inference_ms = round((time.time() - t0) * 1000, 1)

    # Extrair detecções
    detections = []
    class_counts = defaultdict(int)
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({
            "class": cls_name,
            "confidence": round(confidence, 4),
            "bbox": [x1, y1, x2, y2]
        })
        class_counts[cls_name] += 1

    # Regras de negócio
    business = compute_business_rules(detections, img_area)

    # Imagem anotada
    annotated = draw_detections(img, detections)
    annotated_b64 = encode_image(annotated)

    # Atualizar histórico
    total_inspected += 1
    if business["status"] == "reprovado":
        total_defective += 1
        for cls in class_counts:
            session_stats[cls] += class_counts[cls]

    record = {
        "id": total_inspected,
        "timestamp": datetime.now().isoformat(),
        "filename": file.filename,
        "num_defects": len(detections),
        "class_counts": dict(class_counts),
        "defect_area_pct": business["defect_area_pct"],
        "status": business["status"],
        "severity": business["severity"],
        "inference_ms": inference_ms
    }
    inspection_history.append(record)
    if len(inspection_history) > 100:
        inspection_history.pop(0)

    return {
        "success": True,
        "filename": file.filename,
        "image_size": {"width": w, "height": h},
        "inference_ms": inference_ms,
        "detections": detections,
        "class_counts": dict(class_counts),
        "business": business,
        "annotated_image": annotated_b64,
        "stats": {
            "total_inspected": total_inspected,
            "total_defective": total_defective,
            "rejection_rate_pct": round((total_defective / total_inspected) * 100, 1)
        }
    }

@app.get("/stats")
def get_stats():
    """Retorna estatísticas da sessão."""
    rejection_rate = round((total_defective / total_inspected) * 100, 1) if total_inspected > 0 else 0
    return {
        "total_inspected": total_inspected,
        "total_approved": total_inspected - total_defective,
        "total_defective": total_defective,
        "rejection_rate_pct": rejection_rate,
        "defect_class_counts": dict(session_stats),
        "alert": rejection_rate > 20
    }

@app.get("/history")
def get_history(limit: int = 20):
    """Retorna histórico das últimas inspeções."""
    return {
        "total": len(inspection_history),
        "records": inspection_history[-limit:][::-1]
    }

@app.delete("/reset")
def reset_stats():
    """Zera as estatísticas da sessão."""
    global total_inspected, total_defective
    total_inspected = 0
    total_defective = 0
    session_stats.clear()
    inspection_history.clear()
    return {"message": "Estatísticas resetadas com sucesso."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
