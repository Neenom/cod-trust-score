# 🛡️ COD Trust Score

ระบบประเมินสิทธิ์การสั่งซื้อแบบเก็บเงินปลายทาง (COD) รายผู้ใช้ เพื่อลดปัญหา **buyer abuse** (สั่ง COD แล้วไม่รับของ) โดยไม่บล็อกผู้ใช้ทั้งหมดแบบเหมารวม

ทำขึ้นเป็นเดโมสำหรับงานแข่ง Shopee case · **ข้อมูลทั้งหมดเป็นข้อมูลจำลอง (synthetic)**

## แนวคิด

| Tier | ความหมาย | วงเงิน COD | ตีกลับได้ / 90 วัน |
|---|---|---|---|
| A | น่าเชื่อถือสูง | ไม่จำกัด | 3 |
| B | ปานกลาง | ≤ ฿3,000 | 2 |
| C | ต้องระวัง | ≤ ฿1,000 | 1 |
| D | เสี่ยงสูง | งด COD | 0 |

**Flow (Phase 1: Scorecard + LLM)**

```
ข้อมูลผู้ใช้ → Scorecard (กฎ + Bayesian smoothing)
              ├─ เคสชัดเจน → Tier ทันที
              └─ เคสก้ำกึ่ง / จะถูกตัดสิทธิ์ / มีเหตุผลเป็นข้อความ → LLM (Claude)
                                                              ↓
                                     Guardrail (ปรับได้ ±1 ขั้น, ห้ามตัดสิทธิ์เอง, JSON ผิด → fallback)
                                                              ↓
                                                  โควตา COD + log + อุทธรณ์ได้
```

**Phase 2:** เมื่อเก็บผลออเดอร์จริงได้เพียงพอ เทรนโมเดล XGBoost/LightGBM มาแทน Scorecard และยังใช้ LLM อ่านเหตุผลที่เป็นข้อความ

## โครงสร้างโปรเจกต์

```
app.py                 แอป Streamlit (ภาพรวม, ค้นหาผู้ใช้, จำลอง checkout, นโยบาย)
src/generate_data.py   สร้างข้อมูลผู้ใช้จำลอง 1,000 คน (มีกลุ่ม abuse ฝังไว้)
src/scorecard.py       Scorecard แบบกฎ + ตาราง Tier
src/llm_review.py      เรียก Claude + guardrail + decision log (มีโหมด offline)
data/users.csv         ข้อมูลจำลองที่สร้างไว้แล้ว
```

## วิธีรัน

```bash
pip install -r requirements.txt
python -m src.generate_data        # (ไม่บังคับ) สร้างข้อมูลใหม่
streamlit run app.py
```

ใช้ Claude จริง (ไม่บังคับ ถ้าไม่ตั้งค่าจะใช้โหมดจำลองแบบ offline):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# export LLM_MODEL=claude-haiku-4-5   # เปลี่ยนโมเดลได้
```

บน Streamlit Community Cloud ให้ใส่ `ANTHROPIC_API_KEY` ใน **Settings → Secrets** (ดู `.streamlit/secrets.toml.example`)

## ความปลอดภัยและความเป็นธรรม

- ส่งให้ LLM **เฉพาะข้อมูลพฤติกรรม** ไม่มีชื่อ เบอร์ ที่อยู่ เพศ (PDPA + ลด bias)
- ข้อความเหตุผลการตีกลับถูกครอบด้วย tag และระบุว่าเป็นข้อมูล ไม่ใช่คำสั่ง (กัน prompt injection)
- การตีกลับที่เป็นความผิดของร้านหรือขนส่งไม่นับเป็นความผิดของผู้ซื้อ
- temperature = 0 และบันทึกทุกการตัดสินใน `data/decision_log.jsonl`

