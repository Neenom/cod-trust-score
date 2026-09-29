"""COD Trust Score — Streamlit demo (Scorecard + LLM, Phase 1).

Upload a dataset -> every account is scored by the Scorecard -> the LLM
explains each account's score and tier (and may adjust it by one step).

Run:  streamlit run app.py
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.generate_data import generate
from src.llm_review import review
from src.scorecard import TIER_ORDER, needs_llm_review, score_user, tier_policy

DATA = Path(__file__).parent / "data" / "users.csv"
TIER_COLORS = {"A": "#2E7D32", "B": "#1565C0", "C": "#EF6C00", "D": "#C62828"}

# Columns the Scorecard cannot work without
REQUIRED = ["cod_orders", "success_count", "refused_count", "account_age_days",
            "kyc_verified", "shared_device_accounts"]
# Columns that improve the score but can be missing (filled with a neutral default)
OPTIONAL_DEFAULTS = {"refused_90d": 0, "shopeepay_linked": False, "address_changes_90d": 0,
                     "high_value_ratio": 1.0, "last_refusal_reason": ""}
NUMERIC = ["cod_orders", "success_count", "refused_count", "account_age_days",
           "shared_device_accounts", "refused_90d", "address_changes_90d", "high_value_ratio"]
TEMPLATE_COLS = ["user_id"] + REQUIRED + list(OPTIONAL_DEFAULTS)

st.set_page_config(page_title="COD Trust Score", page_icon="🛡️", layout="wide")

# Streamlit Cloud secrets -> env var for the Anthropic SDK
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.getenv("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass


def fmt_limit(limit) -> str:
    if limit is None:
        return "ไม่จำกัด"
    return "งด COD" if limit == 0 else f"฿{limit:,}"


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Validate and clean an uploaded dataset. Raises ValueError if it cannot be scored."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("ไม่พบคอลัมน์ที่จำเป็น: " + ", ".join(missing))

    notes = []
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            notes.append(f"ไม่มีคอลัมน์ `{col}` ใช้ค่าเริ่มต้น {default!r}")
    if "user_id" not in df.columns:
        df.insert(0, "user_id", [f"ROW{i + 1:05d}" for i in range(len(df))])
        notes.append("ไม่มีคอลัมน์ `user_id` สร้างรหัสตามลำดับแถวให้")
    df["user_id"] = df["user_id"].astype(str).str.strip()

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["kyc_verified", "shopeepay_linked"]:
        df[col] = df[col].fillna(False)
    df["last_refusal_reason"] = df["last_refusal_reason"].fillna("").astype(str)
    for col, default in OPTIONAL_DEFAULTS.items():
        if col in NUMERIC:
            df[col] = df[col].fillna(default)

    bad = df[REQUIRED].isna().any(axis=1)
    if bad.any():
        notes.append(f"ข้ามแถวที่ข้อมูลจำเป็นว่างหรือไม่ใช่ตัวเลข {int(bad.sum())} แถว")
        df = df[~bad]
    if df.empty:
        raise ValueError("ไม่มีแถวที่ใช้ประเมินได้")
    for col in NUMERIC:
        if col != "high_value_ratio":
            df[col] = df[col].astype(int)
    return df.reset_index(drop=True), notes


@st.cache_data
def score_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for u in df.to_dict("records"):
        r = score_user(u)
        top = sorted(r.reasons, key=lambda x: abs(x[0]), reverse=True)[:2]
        rows.append({"score": r.score, "tier": r.tier, "borderline": r.borderline,
                     "llm_needed": needs_llm_review(u, r),
                     "top_factors": " · ".join(f"{t} ({p:+d})" for p, t in top)})
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


@st.cache_data
def read_upload(data: bytes, name: str) -> pd.DataFrame:
    from io import BytesIO
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(data))
    return pd.read_csv(BytesIO(data), encoding="utf-8-sig")


# ---------------------------------------------------------------- Sidebar + data
st.sidebar.title("🛡️ COD Trust Score")
st.sidebar.caption("อัปโหลด dataset → Scorecard ให้คะแนน → LLM อธิบายผลรายบัญชี")

uploaded = st.sidebar.file_uploader("อัปโหลด dataset (CSV / Excel)", type=["csv", "xlsx"])
template = pd.read_csv(DATA, nrows=5)[TEMPLATE_COLS] if DATA.exists() else pd.DataFrame(columns=TEMPLATE_COLS)
st.sidebar.download_button("ดาวน์โหลดไฟล์ตัวอย่าง (template)",
                           template.to_csv(index=False).encode("utf-8-sig"),
                           "cod_trust_template.csv", "text/csv")

if uploaded is not None:
    dataset_key = f"{uploaded.name}:{uploaded.size}"
    try:
        df_in, load_notes = prepare(read_upload(uploaded.getvalue(), uploaded.name))
    except Exception as e:
        st.error(f"อ่านไฟล์ `{uploaded.name}` ไม่ได้: {e}")
        st.info("คอลัมน์ที่จำเป็น: " + ", ".join(f"`{c}`" for c in REQUIRED)
                + " · ดาวน์โหลดไฟล์ตัวอย่างได้ที่แถบด้านซ้าย")
        st.stop()
    source_label = f"ไฟล์ที่อัปโหลด: {uploaded.name}"
else:
    dataset_key = "sample"
    df_in, load_notes = prepare(pd.read_csv(DATA) if DATA.exists() else generate())
    source_label = "ข้อมูลตัวอย่าง (synthetic)"

df = score_all(df_in)

# LLM decisions are kept per dataset so switching files starts fresh
if st.session_state.get("dataset_key") != dataset_key:
    st.session_state["dataset_key"] = dataset_key
    st.session_state["llm"] = {}
llm_results: dict = st.session_state["llm"]

has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
page = st.sidebar.radio("หน้า", ["ผลการประเมิน", "รายบัญชี", "ภาพรวม", "จำลองการสั่งซื้อ COD", "นโยบาย Tier"])
use_llm = st.sidebar.toggle("ใช้ Claude จริง", value=has_key, disabled=not has_key,
                            help="ตั้งค่า ANTHROPIC_API_KEY เพื่อเปิดใช้ ถ้าไม่มีจะใช้โหมดจำลอง (offline)")
st.sidebar.info("LLM: " + ("Claude API" if use_llm else "โหมดจำลอง (offline)"))
st.sidebar.caption(f"ชุดข้อมูล: {source_label} · {len(df):,} บัญชี")


def run_review(u: dict) -> dict:
    return review(u, score_user(u), use_real_llm=use_llm)


def final_tier(uid: str, scorecard_tier: str) -> str:
    d = llm_results.get(uid)
    return d["tier"] if d else scorecard_tier


def show_decision(d: dict, scorecard_tier: str) -> None:
    final = tier_policy(d["tier"])
    a, b, c = st.columns(3)
    a.metric("Tier สุดท้าย", d["tier"], None if d["tier"] == scorecard_tier else f"จาก {scorecard_tier}")
    b.metric("ความผิดของ", d.get("fault", "-"))
    c.metric("ตีกลับได้ / 90 วัน", final["refusals_allowed_90d"])
    if d.get("summary"):
        st.info(d["summary"])
    for reason in d["reasons"]:
        st.write("•", reason)
    for n in d["guardrail_notes"]:
        st.warning("Guardrail: " + n)
    with st.expander("ข้อมูลที่ส่งให้ LLM (ไม่มีข้อมูลระบุตัวตน)"):
        st.code(d["prompt"])


# ---------------------------------------------------------------- Results table
if page == "ผลการประเมิน":
    st.title("ผลการประเมินรายบัญชี")
    st.caption(source_label)
    for n in load_notes:
        st.caption("ℹ️ " + n)

    cols = st.columns(len(TIER_ORDER) + 2)
    cols[0].metric("บัญชีทั้งหมด", f"{len(df):,}")
    for col, t in zip(cols[1:], TIER_ORDER):
        col.metric(f"Tier {t}", f"{(df.tier == t).sum():,}")
    cols[-1].metric("LLM อธิบายแล้ว", f"{len(llm_results):,}")

    f1, f2 = st.columns([3, 1])
    tiers = f1.multiselect("กรอง Tier (Scorecard)", TIER_ORDER, default=TIER_ORDER)
    only_flagged = f2.checkbox("เฉพาะเคสที่ควรให้ LLM ตรวจ")
    view = df[df.tier.isin(tiers)]
    if only_flagged:
        view = view[view.llm_needed]

    st.subheader("🤖 ให้ LLM อธิบายคะแนนและ Tier")
    b1, b2 = st.columns([1, 2])
    pending = view[~view.user_id.isin(llm_results)]
    max_default = len(pending) if not use_llm else min(len(pending), 20)
    n = b1.number_input("จำนวนบัญชี (ที่ยังไม่ได้อธิบาย)", 0, max(len(pending), 0),
                        max_default, help="ใช้ Claude จริงจะมีค่าใช้จ่ายต่อบัญชี จึงตั้งค่าเริ่มต้นไว้ 20")
    b2.write("")
    if b2.button(f"อธิบาย {n} บัญชีตามตัวกรองด้านบน", type="primary", disabled=n == 0):
        batch = pending.head(int(n)).to_dict("records")
        bar = st.progress(0.0, text="กำลังให้ LLM วิเคราะห์...")
        with ThreadPoolExecutor(max_workers=4 if use_llm else 1) as pool:
            for i, (u, d) in enumerate(zip(batch, pool.map(run_review, batch)), 1):
                llm_results[u["user_id"]] = d
                bar.progress(i / len(batch), text=f"วิเคราะห์แล้ว {i}/{len(batch)}")
        bar.empty()
        st.success(f"LLM อธิบายเพิ่ม {len(batch)} บัญชี")

    out = view.assign(
        final_tier=[final_tier(u, t) for u, t in zip(view.user_id, view.tier)],
        cod_limit=[fmt_limit(tier_policy(final_tier(u, t))["cod_limit_thb"])
                   for u, t in zip(view.user_id, view.tier)],
        fault=[llm_results.get(u, {}).get("fault", "") for u in view.user_id],
        llm_summary=[llm_results.get(u, {}).get("summary", "") for u in view.user_id],
    )[["user_id", "score", "tier", "final_tier", "cod_limit", "llm_needed", "top_factors",
       "fault", "llm_summary"]]
    st.dataframe(out, width="stretch", hide_index=True, column_config={
        "user_id": "User ID",
        "score": st.column_config.ProgressColumn("คะแนน", min_value=0, max_value=130, format="%d"),
        "tier": "Tier (Scorecard)",
        "final_tier": "Tier สุดท้าย",
        "cod_limit": "วงเงิน COD",
        "llm_needed": st.column_config.CheckboxColumn("ควรให้ LLM ตรวจ"),
        "top_factors": st.column_config.TextColumn("ปัจจัยหลัก (Scorecard)", width="large"),
        "fault": "ความผิดของ",
        "llm_summary": st.column_config.TextColumn("คำอธิบายจาก LLM", width="large"),
    })
    st.download_button("ดาวน์โหลดผลลัพธ์ (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                       "cod_trust_results.csv", "text/csv")

# ---------------------------------------------------------------- Single account
elif page == "รายบัญชี":
    st.title("รายละเอียดรายบัญชี")
    uid = st.selectbox("User ID (พิมพ์เพื่อค้นหาได้)", df.user_id.tolist())
    u = df[df.user_id == uid].iloc[0].to_dict()
    r = score_user(u)

    c1, c2, c3 = st.columns(3)
    c1.metric("คะแนน Scorecard", r.score)
    c2.metric("Tier (Scorecard)", r.tier)
    c3.metric("วงเงิน COD", fmt_limit(tier_policy(r.tier)["cod_limit_thb"]))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("เหตุผลของคะแนน")
        if r.reasons:
            rs = pd.DataFrame(r.reasons, columns=["คะแนน", "เหตุผล"])
            fig = px.bar(rs, x="คะแนน", y="เหตุผล", orientation="h",
                         color=rs["คะแนน"] > 0, color_discrete_map={True: "#2E7D32", False: "#C62828"})
            fig.update_layout(showlegend=False, yaxis_title=None, height=320)
            st.plotly_chart(fig, width="stretch")
        else:
            st.write("ไม่มีปัจจัยบวกหรือลบ คะแนนคงที่ 100")
    with right:
        st.subheader("ข้อมูลผู้ใช้")
        show = {k: u[k] for k in REQUIRED + list(OPTIONAL_DEFAULTS)}
        st.table(pd.Series(show, name="ค่า").astype(str))

    st.subheader("🤖 คำอธิบายจาก LLM")
    st.caption("เคสนี้อยู่ในกลุ่มที่ควรให้ LLM ตรวจ (ใกล้เส้นแบ่ง Tier, จะถูกตัดสิทธิ์ หรือมีเหตุผลการตีกลับ)"
               if needs_llm_review(u, r) else
               "เคสชัดเจน Scorecard ตัดสินได้เอง แต่ยังให้ LLM อธิบายคะแนนได้")
    if st.button("ให้ LLM อธิบาย" if uid not in llm_results else "ให้ LLM วิเคราะห์ใหม่"):
        with st.spinner("กำลังวิเคราะห์..."):
            llm_results[uid] = run_review(u)
    if uid in llm_results:
        show_decision(llm_results[uid], r.tier)

# ---------------------------------------------------------------- Overview
elif page == "ภาพรวม":
    st.title("ภาพรวมระบบ COD Trust Score")
    st.caption(source_label)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ผู้ใช้ทั้งหมด", f"{len(df):,}")
    c2.metric("งด COD (Tier D)", f"{(df.tier == 'D').sum():,}")
    c3.metric("ควรให้ LLM ตรวจ", f"{df.llm_needed.mean():.0%}")
    base_rate = df.refused_count.sum() / max(df.cod_orders.sum(), 1)
    kept = df[df.tier != "D"]
    new_rate = kept.refused_count.sum() / max(kept.cod_orders.sum(), 1)
    c4.metric("อัตราตีกลับ (ถ้าตัด Tier D)", f"{new_rate:.1%}", f"{(new_rate - base_rate):.1%}",
              delta_color="inverse")

    left, right = st.columns(2)
    counts = df.tier.value_counts().reindex(TIER_ORDER, fill_value=0).reset_index()
    counts.columns = ["tier", "users"]
    left.plotly_chart(px.bar(counts, x="tier", y="users", color="tier",
                             color_discrete_map=TIER_COLORS, title="จำนวนผู้ใช้แต่ละ Tier"),
                      width="stretch")
    right.plotly_chart(px.histogram(df, x="score", color="tier", nbins=30,
                                    color_discrete_map=TIER_COLORS,
                                    category_orders={"tier": TIER_ORDER},
                                    title="การกระจายของคะแนน"),
                       width="stretch")

    # Only synthetic data has a ground-truth label to evaluate against
    if "segment_truth" in df.columns:
        st.subheader("ระบบจับกลุ่ม abuse ได้แม่นแค่ไหน (เทียบกับ ground truth ของข้อมูลจำลอง)")
        ct = pd.crosstab(df.segment_truth, df.tier).reindex(columns=TIER_ORDER, fill_value=0)
        st.dataframe(ct, width="stretch")
        caught = (df[df.segment_truth == "abuser"].tier.isin(["C", "D"])).mean()
        fp = (df[df.segment_truth == "loyal"].tier.isin(["C", "D"])).mean()
        st.caption(f"ผู้ใช้สาย abuse ถูกจัดเป็น Tier C/D: **{caught:.0%}** · "
                   f"ลูกค้าประจำถูกจัดผิดเป็น C/D (false positive): **{fp:.0%}**")

# ---------------------------------------------------------------- Checkout sim
elif page == "จำลองการสั่งซื้อ COD":
    st.title("จำลองหน้า Checkout")
    c1, c2 = st.columns(2)
    uid = c1.selectbox("ผู้ใช้", df.user_id.tolist())
    amount = c2.number_input("ยอดสั่งซื้อ (บาท)", 50, 50000, 1500, step=100)
    u = df[df.user_id == uid].iloc[0].to_dict()
    r = score_user(u)
    tier = final_tier(uid, r.tier)
    pol = tier_policy(tier)
    refusals_left = pol["refusals_allowed_90d"] - int(u["refused_90d"])

    if st.button("เลือกชำระเงินปลายทาง (COD)", type="primary"):
        limit = pol["cod_limit_thb"]
        if limit == 0:
            st.error("❌ บัญชีนี้ยังใช้ COD ไม่ได้ กรุณาชำระผ่าน ShopeePay / บัตร / โอนเงิน")
            st.caption("รับของสำเร็จต่อเนื่องเพื่อปลดล็อก COD อีกครั้ง หรือยื่นอุทธรณ์ได้")
        elif refusals_left <= 0:
            st.error("❌ ใช้สิทธิ์ตีกลับครบแล้วในรอบ 90 วัน ใช้ COD ได้อีกครั้งเมื่อครบรอบ")
        elif limit is not None and amount > limit:
            st.warning(f"⚠️ ยอดเกินวงเงิน COD ของคุณ ({fmt_limit(limit)}) กรุณาชำระล่วงหน้า")
        else:
            st.success(f"✅ อนุญาต COD · Tier {tier} · ตีกลับได้อีก {refusals_left} ครั้งใน 90 วัน")
    src = "หลัง LLM ตรวจ" if uid in llm_results else "จาก Scorecard"
    st.caption(f"Tier {tier} ({pol['label']}, {src}) · วงเงิน {fmt_limit(pol['cod_limit_thb'])} · คะแนน {r.score}")

# ---------------------------------------------------------------- Policy
else:
    st.title("นโยบาย Tier")
    st.table(pd.DataFrame([{**tier_policy(t), "cod_limit_thb": fmt_limit(tier_policy(t)["cod_limit_thb"])}
                           for t in TIER_ORDER]).rename(columns={
        "tier": "Tier", "cod_limit_thb": "วงเงิน COD", "refusals_allowed_90d": "ตีกลับได้/90 วัน",
        "label": "ความหมาย"}))
    st.markdown(f"""
**Flow การตัดสิน**
1. อัปโหลด dataset → Scorecard ให้คะแนนจากพฤติกรรม (โค้ด ไม่ใช่ AI) ทุกบัญชี
2. LLM อธิบายว่าทำไมแต่ละบัญชีได้คะแนนและ Tier นี้ และแยกว่าการตีกลับเป็นความผิดของใคร
3. Guardrail: LLM ปรับได้ไม่เกิน ±1 ขั้น, ห้ามตัดสิทธิ์ COD เอง, JSON ผิด → ใช้ผล Scorecard
4. บันทึก log ทุกการตัดสิน · ผู้ใช้อุทธรณ์ได้ · รับของสำเร็จแล้วคะแนนฟื้น

**คอลัมน์ที่ต้องมีใน dataset:** {", ".join(f"`{c}`" for c in REQUIRED)}
**คอลัมน์เสริม:** {", ".join(f"`{c}`" for c in OPTIONAL_DEFAULTS)} (ถ้าไม่มีจะใช้ค่าเริ่มต้น)

**Phase 2:** เมื่อเก็บผลออเดอร์จริงได้พอ เทรนโมเดล XGBoost มาแทน Scorecard
""")
