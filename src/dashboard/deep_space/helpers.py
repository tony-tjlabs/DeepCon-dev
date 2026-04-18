"""
Deep Space 헬퍼 함수
====================
예측, 이상 탐지, 임베딩 계산 등 공용 함수.
torch/sklearn은 함수 내부에서 lazy import.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.styles import COLORS, RADIUS, SPACING

# ─── 상수 ──────────────────────────────────────────────────────
MAX_SEQ_LEN = 50          # 예측 시 사용할 최근 시퀀스 길이
MIN_SEQ_LEN = 3           # 유효 시퀀스 최소 길이
SPECIAL_TOKEN_RANGE = 5   # 특수 토큰 ID (0~4: PAD, UNK, CLS, SEP, MASK)
ANOMALY_SIGMA = 1.5       # 이상치 판정 기본 sigma 배수
BATCH_DEFAULT_SIZE = 256  # 배치 추론 기본 크기


def extract_worker_sequences(
    journey_df: pd.DataFrame,
    work_hour_only: bool = True,
) -> dict[str, list[str]]:
    """journey.parquet에서 작업자별 locus 시퀀스 추출.

    Args:
        journey_df: journey.parquet DataFrame
        work_hour_only: True면 is_work_hour=True만 사용 (야간 헬멧 노이즈 제거)
    """
    from src.dashboard.deep_space.model_loader import detect_time_column

    if journey_df is None or journey_df.empty:
        return {}

    locus_col = "locus_id" if "locus_id" in journey_df.columns else None
    if locus_col is None:
        for col in ["locus", "corrected_locus_id"]:
            if col in journey_df.columns:
                locus_col = col
                break
    if locus_col is None:
        return {}

    user_col = "user_no" if "user_no" in journey_df.columns else None
    if user_col is None:
        return {}

    # ★ 작업시간 필터 — 야간 헬멧 노이즈 제거 (45%+ 비작업시간 데이터)
    df = journey_df
    if work_hour_only and "is_work_hour" in df.columns:
        df = df[df["is_work_hour"] == True]

    time_col = detect_time_column(df)

    sequences = {}
    for user_no, group in df.groupby(user_col):
        if time_col:
            group = group.sort_values(time_col)
        loci = group[locus_col].dropna().tolist()
        # 연속 중복 제거 (같은 장소 연속 체류 -> 1개로)
        deduped = []
        for loc in loci:
            if not deduped or deduped[-1] != loc:
                deduped.append(str(loc))
        if len(deduped) >= MIN_SEQ_LEN:
            sequences[str(user_no)] = deduped
    return sequences


def predict_next(model, tokenizer, sequence: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    """시퀀스의 다음 위치 Top-K 예측."""
    import torch

    cfg = tokenizer.config

    # 토큰화: [CLS] + seq(최근) + [MASK] (special_tokens 없이 수동 구성)
    recent = sequence[-MAX_SEQ_LEN:]
    token_ids = tokenizer.encode(recent, add_special_tokens=False)
    token_ids = [cfg.cls_token_id] + token_ids + [cfg.mask_token_id]

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"]

    # 마지막 위치 ([MASK])만 softmax (전체 seq 대비 연산 절감)
    import torch.nn.functional as F
    mask_probs = F.softmax(logits[0, -1, :], dim=-1).clone()  # [vocab_size]

    # 특수 토큰 제외
    for special_id in range(SPECIAL_TOKEN_RANGE):
        mask_probs[special_id] = 0.0

    # Top-K
    topk_vals, topk_ids = torch.topk(mask_probs, top_k)
    results = []
    for val, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
        decoded = tokenizer.decode([idx], skip_special_tokens=False)
        locus_name = decoded[0] if decoded else f"token_{idx}"
        results.append((locus_name, val))
    return results


def compute_anomaly_score(
    model, tokenizer, sequence: list[str], mask_chunk_size: int = 8,
) -> tuple[float, list[tuple[int, str, float]]]:
    """시퀀스의 이상 점수 계산 (chunk batching). 각 위치의 예측 확률 기반."""
    import torch
    import torch.nn.functional as F

    if len(sequence) < MIN_SEQ_LEN:
        return 0.0, []

    cfg = tokenizer.config

    # 전체 시퀀스 토큰화 (CLS + loci + SEP)
    token_ids = tokenizer.encode(sequence, add_special_tokens=True)

    # 마스킹 대상: CLS(0)과 SEP(마지막) 제외
    locus_positions = list(range(1, len(token_ids) - 1))
    if not locus_positions:
        return 0.0, []

    surprisals = []

    # chunk 단위로 배치 처리 (mask_chunk_size개 위치씩)
    with torch.no_grad():
        for chunk_start in range(0, len(locus_positions), mask_chunk_size):
            chunk_positions = locus_positions[chunk_start: chunk_start + mask_chunk_size]

            # 각 위치별 마스킹된 시퀀스 생성
            batch_ids = []
            for pos in chunk_positions:
                masked = list(token_ids)
                masked[pos] = cfg.mask_token_id
                batch_ids.append(masked)

            input_tensor = torch.tensor(batch_ids, dtype=torch.long)
            attn_mask = torch.ones_like(input_tensor)

            outputs = model(input_tensor, attn_mask)
            logits = outputs["logits"]  # [chunk, seq_len, vocab_size]

            for batch_i, pos in enumerate(chunk_positions):
                original_id = token_ids[pos]
                probs = F.softmax(logits[batch_i, pos], dim=-1)
                prob_of_actual = probs[original_id].item()
                surprisal = -np.log(max(prob_of_actual, 1e-10))
                decoded = tokenizer.decode([original_id], skip_special_tokens=False)
                locus_name = decoded[0] if decoded else f"token_{original_id}"
                surprisals.append((pos - 1, locus_name, surprisal))

    if not surprisals:
        return 0.0, []

    avg_surprisal = np.mean([s[2] for s in surprisals])
    # 이상 위치: 평균 + 1.5 표준편차 이상
    std = np.std([s[2] for s in surprisals])
    threshold = avg_surprisal + ANOMALY_SIGMA * std
    anomalies = [(i, loc, score) for i, loc, score in surprisals if score > threshold]

    return avg_surprisal, anomalies


def compute_locus_embeddings(model, tokenizer) -> pd.DataFrame:
    """모델의 Locus 임베딩 추출 + 2D 투영 (session_state 캐싱)."""
    cache_key = "_ds_locus_embeddings"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    import torch
    from sklearn.decomposition import PCA

    # 임베딩 레이어에서 직접 추출
    with torch.no_grad():
        embeddings = model.locus_embedding.weight.detach().cpu().numpy()  # [vocab_size, d_model]

    # 특수 토큰(0~4) 제외
    locus_names = []
    locus_vectors = []
    for token_id in range(5, embeddings.shape[0]):
        name = tokenizer.decode([token_id], skip_special_tokens=False)[0]
        if name and name != "[UNK]":
            locus_names.append(name)
            locus_vectors.append(embeddings[token_id])

    if len(locus_vectors) < 3:
        return pd.DataFrame()

    vectors = np.array(locus_vectors)

    # PCA 2D
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(vectors)

    df = pd.DataFrame({
        "locus_id": locus_names,
        "x": coords_2d[:, 0],
        "y": coords_2d[:, 1],
        "pca_var_explained": [pca.explained_variance_ratio_.sum()] * len(locus_names),
    })

    st.session_state[cache_key] = df
    return df


def compute_similarity_matrix(model, tokenizer) -> tuple[pd.DataFrame, list[str]]:
    """Locus 간 코사인 유사도 매트릭스 (session_state 캐싱)."""
    cache_key = "_ds_similarity_matrix"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    import torch

    with torch.no_grad():
        embeddings = model.locus_embedding.weight.detach().cpu().numpy()

    locus_names = []
    locus_vectors = []
    for token_id in range(5, embeddings.shape[0]):
        name = tokenizer.decode([token_id], skip_special_tokens=False)[0]
        if name and name != "[UNK]":
            locus_names.append(name)
            locus_vectors.append(embeddings[token_id])

    vectors = np.array(locus_vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-10)
    sim_matrix = normalized @ normalized.T

    df = pd.DataFrame(sim_matrix, index=locus_names, columns=locus_names)
    result = (df, locus_names)
    st.session_state[cache_key] = result
    return result


def classify_confidence(prob: float) -> tuple[str, str]:
    """예측 확률 -> (라벨, badge 종류).

    Phase 1 합의:
        >= 0.4  -> High (success)
        0.2~0.4 -> Medium (warning)
        < 0.2   -> Low (danger)
    """
    if prob >= 0.4:
        return "High", "success"
    elif prob >= 0.2:
        return "Medium", "warning"
    else:
        return "Low", "danger"


def classify_severity(
    score: float, mean: float, std: float, sigma_mult: float = 1.5,
) -> tuple[str, str]:
    """Surprisal -> (위험도 라벨, COLORS 키).

    Phase 1 합의 (4단계):
        >= mean + 2.5*std  -> Critical (danger)
        >= mean + 2.0*std  -> High (confined)
        >= mean + 1.5*std  -> Medium (warning)
        나머지             -> Low (text_muted)

    sigma_mult 파라미터는 슬라이더 조절용 (기본 분류 기준은 고정).
    """
    if score >= mean + 2.5 * std:
        return "Critical", "danger"
    elif score >= mean + 2.0 * std:
        return "High", "confined"
    elif score >= mean + sigma_mult * std:
        return "Medium", "warning"
    else:
        return "Low", "text_muted"


def get_inflow_outflow(
    matrix: np.ndarray,
    locus_ids: list[str],
    target: str,
    top_k: int = 5,
) -> dict:
    """특정 Locus의 유입/유출 Top-K.

    Returns:
        {"inflow": [(locus_id, prob), ...], "outflow": [(locus_id, prob), ...]}
    """
    if target not in locus_ids:
        return {"inflow": [], "outflow": []}

    idx = locus_ids.index(target)

    # 유출: target -> 다른 Locus (해당 행)
    outflow_row = matrix[idx, :]
    outflow_indices = np.argsort(outflow_row)[::-1]
    outflow = []
    for j in outflow_indices:
        if locus_ids[j] != target and outflow_row[j] > 0.001:
            outflow.append((locus_ids[j], float(outflow_row[j])))
            if len(outflow) >= top_k:
                break

    # 유입: 다른 Locus -> target (해당 열)
    inflow_col = matrix[:, idx]
    inflow_indices = np.argsort(inflow_col)[::-1]
    inflow = []
    for i in inflow_indices:
        if locus_ids[i] != target and inflow_col[i] > 0.001:
            inflow.append((locus_ids[i], float(inflow_col[i])))
            if len(inflow) >= top_k:
                break

    return {"inflow": inflow, "outflow": outflow}


def render_card(title: str, content: str):
    """정보 카드."""
    st.markdown(
        f"<div style='background:{COLORS['card_bg']}; border:1px solid {COLORS['border']}; "
        f"border-radius:{RADIUS['md']}; padding:{SPACING['lg']}; margin-bottom:{SPACING['md']};'>"
        f"<div style='color:{COLORS['text_muted']}; font-size:0.8rem; margin-bottom:4px;'>{title}</div>"
        f"<div style='color:{COLORS['text']}; font-size:1.1rem; font-weight:600;'>{content}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _batch_cache_key(sequences: list[list[str]], top_k: int) -> str | None:
    """배치 예측 캐시 키 생성. 대규모 입력은 hash로 축약.

    날짜 변경 시 캐시 오염 방지: 현재 분석 날짜 + 시퀀스 길이 합산 + 해시 20자.
    """
    import hashlib
    try:
        # 날짜 정보를 캐시 키에 포함 (날짜 변경 시 캐시 무효화)
        date_str = ""
        if "current_date" in st.session_state:
            date_str = str(st.session_state["current_date"])

        # 시퀀스 총 길이 합산 (추가 핑거프린트)
        total_len = sum(len(seq) for seq in sequences)

        fingerprint_parts = [str(top_k), str(len(sequences)), str(total_len), date_str]
        for seq in sequences:
            tail = "|".join(seq[-3:]) if seq else ""
            fingerprint_parts.append(f"{len(seq)}:{tail}")
        raw = "\n".join(fingerprint_parts)
        h = hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()[:20]
        return f"_ds_batch_pred_{len(sequences)}_{top_k}_{h}"
    except Exception:
        return None


def predict_next_batch(
    model,
    tokenizer,
    sequences: list[list[str]],
    batch_size: int = 256,
    top_k: int = 3,
    use_cache: bool = True,
) -> list[list[tuple[str, float]]]:
    """N명의 시퀀스를 배치로 묶어 한 번에 추론.

    Args:
        model: Deep Space Transformer 모델
        tokenizer: 토크나이저
        sequences: N명의 locus_id 시퀀스 리스트
        batch_size: 배치 크기 (GPU 없는 CPU는 128~256 권장)
        top_k: 각 worker당 반환할 예측 개수
        use_cache: True면 session_state 캐시 활용 (동일 시퀀스+top_k → 재추론 생략)

    Returns:
        N개 리스트, 각 원소는 [(locus_id, probability), ...] (top_k개)
        예측 불가(빈 시퀀스)이면 해당 원소 = []
    """
    import torch

    if model is None or tokenizer is None or not sequences:
        return [[] for _ in sequences]

    # session_state 캐시 조회 (탭 간 공유)
    if use_cache:
        cache_key = _batch_cache_key(sequences, top_k)
        if cache_key and cache_key in st.session_state:
            cached = st.session_state[cache_key]
            if len(cached) == len(sequences):
                return cached

    cfg = tokenizer.config
    max_seq_len = MAX_SEQ_LEN

    results: list[list[tuple[str, float]]] = [[] for _ in sequences]
    valid_indices: list[int] = []
    valid_token_lists: list[list[int]] = []

    # 1. 각 시퀀스를 토큰화 (빈 시퀀스는 건너뜀)
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        recent = seq[-max_seq_len:]
        token_ids = tokenizer.encode(recent, add_special_tokens=False)
        # [CLS] + tokens + [MASK]
        token_ids = [cfg.cls_token_id] + token_ids + [cfg.mask_token_id]
        valid_indices.append(i)
        valid_token_lists.append(token_ids)

    if not valid_token_lists:
        return results

    # 2. 배치 단위로 처리
    for batch_start in range(0, len(valid_token_lists), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_token_lists))
        batch_tokens = valid_token_lists[batch_start:batch_end]
        batch_indices = valid_indices[batch_start:batch_end]

        # 왼쪽 패딩 (짧은 시퀀스는 왼쪽을 PAD로 채움)
        max_len = max(len(t) for t in batch_tokens)
        padded_batch = []
        attention_masks = []

        for tokens in batch_tokens:
            pad_len = max_len - len(tokens)
            padded = [cfg.pad_token_id] * pad_len + tokens
            mask = [0] * pad_len + [1] * len(tokens)
            padded_batch.append(padded)
            attention_masks.append(mask)

        input_ids = torch.tensor(padded_batch, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long)

        # 3. 배치 forward (softmax는 mask 위치만 적용 — 전체 seq 대비 연산 절감)
        import torch.nn.functional as F
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]  # [batch, seq_len, vocab_size]

        # 4. 각 시퀀스의 MASK 위치(마지막)에서 softmax 추출
        for batch_idx, orig_idx in enumerate(batch_indices):
            mask_probs = F.softmax(logits[batch_idx, -1, :], dim=-1).clone()  # [vocab_size]

            # 특수 토큰 제외 (PAD, UNK, CLS, SEP, MASK)
            for special_id in range(SPECIAL_TOKEN_RANGE):
                mask_probs[special_id] = 0.0

            # Top-K
            topk_vals, topk_ids = torch.topk(mask_probs, top_k)
            preds = []
            for val, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                decoded = tokenizer.decode([idx], skip_special_tokens=False)
                locus_name = decoded[0] if decoded else f"token_{idx}"
                preds.append((locus_name, val))
            results[orig_idx] = preds

    # session_state 캐시 저장
    if use_cache:
        cache_key = _batch_cache_key(sequences, top_k)
        if cache_key:
            st.session_state[cache_key] = results

    return results
