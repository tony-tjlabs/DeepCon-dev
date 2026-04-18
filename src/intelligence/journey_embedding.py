"""
Journey 임베딩 + 클러스터링 모델
===================================
작업자별 Locus 이동 시퀀스를 Word2Vec으로 임베딩하고,
K-means로 Journey 패턴을 클러스터링.

핵심 개념:
  "Journey = 문장(Sentence), Locus = 단어(Word)"
  - 각 작업자의 하루 이동 경로 = 토큰 시퀀스
  - Word2Vec(Skip-gram) 학습 → Locus 의미 벡터 획득
  - 작업자별 평균 벡터 → K-means 클러스터링 → 패턴 분류

패턴 유형 (예시, 자동 명명):
  Cluster-0: FAB 중심 작업자   → EWI 높음, FAB 위주
  Cluster-1: 이동형 작업자     → 넓은 범위 이동
  Cluster-2: 외곽 작업자       → 야외/지원 구역 중심
  Cluster-3: 단기 체류 작업자  → 짧은 근무, 이동 적음

사용법:
    # 학습 (Dev 환경, journey.parquet 필요)
    emb = JourneyEmbedder(sector_id)
    emb.train_from_parquets(processed_dates)
    emb.save()

    # 추론 (Cloud 포함)
    emb = JourneyEmbedder(sector_id).load()
    worker_df = emb.assign_clusters(worker_df, journey_df)

★ 의존성: gensim(Word2Vec), scikit-learn(KMeans, PCA/UMAP)
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)

# ─── 하이퍼파라미터 ────────────────────────────────────────────────
W2V_DIM        = 64     # 임베딩 차원
W2V_WINDOW     = 5      # Skip-gram 컨텍스트 창
W2V_MIN_COUNT  = 1      # 최소 토큰 등장 횟수
W2V_EPOCHS     = 30     # 학습 에폭
N_CLUSTERS     = 5      # K-means 클러스터 수 (조정 가능)
RANDOM_STATE   = 42


class JourneyEmbedder:
    """
    Journey Word2Vec 임베딩 + K-means 클러스터링 모델.

    파일 구조 (data/model/{sector_id}/):
        w2v_model.pkl          ← Word2Vec 모델 (gensim)
        kmeans_model.pkl       ← KMeans 모델 (scikit-learn)
        cluster_labels.json    ← 클러스터 ID → 자동 명칭 + 대표 Locus
        embedder_meta.json     ← 학습 메타 (날짜, 작업자 수, etc.)
    """

    def __init__(self, sector_id: str = cfg.SECTOR_ID):
        self.sector_id  = sector_id
        self.model_dir  = cfg.MODEL_DIR / sector_id
        self.w2v_model  = None
        self.kmeans     = None
        self.cluster_labels: dict[int, dict] = {}  # {cluster_id: {name, loci, color}}
        self._is_trained = False

    # ─── 학습 ────────────────────────────────────────────────────
    def train_from_parquets(
        self,
        processed_dates: list[str],
        n_clusters: int = N_CLUSTERS,
        verbose: bool = True,
    ) -> "JourneyEmbedder":
        """
        journey.parquet 파일들로 Word2Vec + K-means 학습.

        ★ Dev 환경 전용: journey.parquet(35MB/일) 로드 필요
        """
        try:
            from gensim.models import Word2Vec
        except ImportError:
            raise ImportError("gensim 설치 필요: pip install gensim")

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("scikit-learn 설치 필요: pip install scikit-learn")

        from src.pipeline.cache_manager import _date_dir

        # ── Step 1: 작업자별 Locus 시퀀스 수집 ──────────────────
        sentences: list[list[str]] = []  # Word2Vec 입력
        worker_seqs: dict[str, list[str]] = {}  # 작업자별 전체 시퀀스

        for date_str in processed_dates:
            journey_path = _date_dir(date_str, self.sector_id) / "journey.parquet"
            if not journey_path.exists():
                logger.warning(f"journey.parquet 없음: {date_str}")
                continue

            try:
                jdf = pd.read_parquet(
                    journey_path,
                    columns=["user_no", "locus_id", "seq"],
                )
            except Exception as e:
                logger.warning(f"journey.parquet 로드 실패 ({date_str}): {e}")
                continue

            if "locus_id" not in jdf.columns:
                continue

            # 시간순 정렬 → 작업자별 Locus 시퀀스
            jdf = jdf.sort_values(["user_no", "seq"])
            for user_no, grp in jdf.groupby("user_no"):
                tokens = (
                    grp["locus_id"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )
                if len(tokens) < 3:
                    continue
                sentences.append(tokens)
                key = str(user_no)
                worker_seqs.setdefault(key, []).extend(tokens)

        if not sentences:
            raise ValueError("학습 가능한 Journey 데이터 없음")

        if verbose:
            logger.info(f"Word2Vec 학습: {len(sentences)}개 시퀀스, "
                        f"{len(set(t for s in sentences for t in s))}개 유니크 Locus")

        # ── Step 2: Word2Vec 학습 ─────────────────────────────────
        self.w2v_model = Word2Vec(
            sentences   = sentences,
            vector_size = W2V_DIM,
            window      = W2V_WINDOW,
            min_count   = W2V_MIN_COUNT,
            sg          = 1,          # Skip-gram
            epochs      = W2V_EPOCHS,
            seed        = RANDOM_STATE,
            workers     = 4,
        )

        # ── Step 3: 작업자별 평균 임베딩 벡터 ─────────────────────
        user_vectors, user_ids = self._build_user_vectors(worker_seqs)

        if len(user_vectors) < n_clusters:
            logger.warning(
                f"작업자({len(user_vectors)}) < 클러스터({n_clusters}) → "
                f"n_clusters를 {len(user_vectors)}로 조정"
            )
            n_clusters = max(2, len(user_vectors))

        # ── Step 4: K-means 클러스터링 ───────────────────────────
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = self.kmeans.fit_predict(user_vectors)

        # ── Step 5: 클러스터 자동 명칭 생성 ──────────────────────
        self.cluster_labels = self._auto_name_clusters(
            user_vectors, user_ids, labels, n_clusters
        )

        self._is_trained = True
        logger.info(f"학습 완료: {n_clusters}개 클러스터, {len(user_vectors)}명 작업자")
        return self

    def _build_user_vectors(
        self,
        worker_seqs: dict[str, list[str]],
    ) -> tuple[np.ndarray, list[str]]:
        """작업자별 평균 임베딩 벡터 계산."""
        vectors, ids = [], []
        wv = self.w2v_model.wv
        for user_no, tokens in worker_seqs.items():
            vecs = [wv[t] for t in tokens if t in wv]
            if not vecs:
                continue
            vectors.append(np.mean(vecs, axis=0))
            ids.append(user_no)
        return np.array(vectors), ids

    def _auto_name_clusters(
        self,
        user_vectors: np.ndarray,
        user_ids: list[str],
        labels: np.ndarray,
        n_clusters: int,
    ) -> dict[int, dict]:
        """
        클러스터별 대표 Locus + 자동 명칭 생성.
        centroid와 가장 가까운 Locus 토큰을 찾아 명칭화.
        """
        result: dict[int, dict] = {}
        wv = self.w2v_model.wv

        # 클러스터 크기 기준 색상
        cluster_colors = [
            "#00AEEF", "#00C897", "#FFB300", "#FF8C42", "#FF4C4C",
            "#9B59B6", "#3498DB", "#2ECC71", "#E67E22", "#E74C3C",
        ]

        for cid in range(n_clusters):
            mask    = labels == cid
            centroid = self.kmeans.cluster_centers_[cid]
            size    = int(mask.sum())

            # centroid와 가장 유사한 Locus 토큰 (상위 3개)
            try:
                similar = wv.similar_by_vector(centroid, topn=3)
                top_loci = [s[0] for s in similar]
            except Exception:
                top_loci = []

            # 자동 명칭: 상위 Locus 기반
            if top_loci:
                name = f"패턴 {chr(65 + cid)}: {'/'.join(top_loci[:2])}"
            else:
                name = f"패턴 {chr(65 + cid)}"

            result[cid] = {
                "name":   name,
                "loci":   top_loci,
                "size":   size,
                "color":  cluster_colors[cid % len(cluster_colors)],
            }

        return result

    # ─── 추론 ────────────────────────────────────────────────────
    def assign_clusters(
        self,
        worker_df: pd.DataFrame,
        journey_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        worker_df에 cluster_id, cluster_name 컬럼 추가.

        Args:
            worker_df:  worker.parquet (user_no 컬럼 필요)
            journey_df: journey.parquet (user_no, locus_id 컬럼 필요)

        반환: worker_df에 cluster_id(int), cluster_name(str) 추가된 복사본
        """
        if not self._is_trained or self.w2v_model is None or self.kmeans is None:
            raise RuntimeError("모델이 학습/로드되지 않았습니다. load() 또는 train() 먼저 호출하세요.")

        wv = self.w2v_model.wv

        # 작업자별 시퀀스 → 평균 벡터
        seqs: dict[str, list[str]] = {}
        jdf = journey_df[["user_no", "locus_id"]].dropna()
        for user_no, grp in jdf.groupby("user_no"):
            seqs[str(user_no)] = grp["locus_id"].astype(str).tolist()

        rows = []
        for user_no, tokens in seqs.items():
            vecs = [wv[t] for t in tokens if t in wv]
            if not vecs:
                rows.append({"user_no": user_no, "cluster_id": -1, "cluster_name": "미분류"})
                continue
            vec   = np.mean(vecs, axis=0).reshape(1, -1)
            cid   = int(self.kmeans.predict(vec)[0])
            cname = self.cluster_labels.get(cid, {}).get("name", f"패턴 {cid}")
            rows.append({"user_no": user_no, "cluster_id": cid, "cluster_name": cname})

        cluster_df = pd.DataFrame(rows)
        worker_df  = worker_df.copy()

        # user_no 타입 통일
        cluster_df["user_no"] = cluster_df["user_no"].astype(str)
        worker_df["user_no"]  = worker_df["user_no"].astype(str)

        return worker_df.merge(cluster_df, on="user_no", how="left")

    def get_2d_projection(
        self,
        journey_df: pd.DataFrame,
        method: str = "pca",
        n_components: int = 2,
    ) -> pd.DataFrame:
        """
        작업자 임베딩 벡터를 2D/3D로 투영 (시각화용).

        Args:
            method: "pca" 또는 "umap"
            n_components: 차원 수 (2 또는 3)

        반환: DataFrame(user_no, x, y, [z,] cluster_id, cluster_name)
        """
        if not self._is_trained:
            raise RuntimeError("모델 미학습")

        n_components = min(n_components, 3)  # 최대 3차원

        wv = self.w2v_model.wv
        seqs: dict[str, list[str]] = {}
        jdf = journey_df[["user_no", "locus_id"]].dropna()
        for user_no, grp in jdf.groupby("user_no"):
            seqs[str(user_no)] = grp["locus_id"].astype(str).tolist()

        ids, vecs = [], []
        for user_no, tokens in seqs.items():
            v = [wv[t] for t in tokens if t in wv]
            if v:
                ids.append(user_no)
                vecs.append(np.mean(v, axis=0))

        if not vecs:
            cols = ["user_no", "x", "y"] + (["z"] if n_components >= 3 else []) + ["cluster_id"]
            return pd.DataFrame(columns=cols)

        arr = np.array(vecs)

        if method == "umap":
            try:
                from umap import UMAP
                proj = UMAP(n_components=n_components, random_state=RANDOM_STATE).fit_transform(arr)
            except ImportError:
                logger.warning("umap-learn 미설치 -> PCA로 대체")
                from sklearn.decomposition import PCA
                proj = PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(arr)
        else:
            from sklearn.decomposition import PCA
            proj = PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(arr)

        cluster_ids = self.kmeans.predict(arr)
        data = {
            "user_no":    ids,
            "x":          proj[:, 0],
            "y":          proj[:, 1],
        }
        if n_components >= 3:
            data["z"] = proj[:, 2]
        data["cluster_id"] = cluster_ids.astype(int)

        df = pd.DataFrame(data)
        df["cluster_name"] = df["cluster_id"].map(
            lambda c: self.cluster_labels.get(c, {}).get("name", f"패턴 {c}")
        )
        return df

    # ─── 저장 / 로드 ─────────────────────────────────────────────
    def save(self) -> None:
        """모델을 data/model/{sector_id}/ 에 저장."""
        if not self._is_trained:
            raise RuntimeError("저장할 모델 없음")

        self.model_dir.mkdir(parents=True, exist_ok=True)

        with open(self.model_dir / "w2v_model.pkl", "wb") as f:
            pickle.dump(self.w2v_model, f)

        with open(self.model_dir / "kmeans_model.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)

        with open(self.model_dir / "cluster_labels.json", "w", encoding="utf-8") as f:
            json.dump(self.cluster_labels, f, ensure_ascii=False, indent=2, default=str)

        meta = {
            "sector_id":    self.sector_id,
            "w2v_dim":      W2V_DIM,
            "n_clusters":   len(self.cluster_labels),
            "vocab_size":   len(self.w2v_model.wv),
        }
        with open(self.model_dir / "embedder_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"JourneyEmbedder 저장 완료: {self.model_dir}")

    def load(self) -> "JourneyEmbedder":
        """저장된 모델 로드."""
        w2v_path    = self.model_dir / "w2v_model.pkl"
        kmeans_path = self.model_dir / "kmeans_model.pkl"
        labels_path = self.model_dir / "cluster_labels.json"

        if not w2v_path.exists():
            raise FileNotFoundError(f"W2V 모델 없음: {w2v_path}")

        with open(w2v_path, "rb") as f:
            self.w2v_model = pickle.load(f)

        with open(kmeans_path, "rb") as f:
            self.kmeans = pickle.load(f)

        with open(labels_path, encoding="utf-8") as f:
            raw = json.load(f)
            # JSON key는 str이므로 int로 변환
            self.cluster_labels = {int(k): v for k, v in raw.items()}

        self._is_trained = True
        logger.info(f"JourneyEmbedder 로드 완료: {self.model_dir}")
        return self

    def is_available(self) -> bool:
        """저장된 모델이 존재하는지 확인."""
        return (self.model_dir / "w2v_model.pkl").exists()

    @property
    def vocab(self) -> list[str]:
        """학습된 Locus 토큰 목록."""
        if self.w2v_model is None:
            return []
        return list(self.w2v_model.wv.key_to_index.keys())

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_labels)


# ─── 편의 함수 ────────────────────────────────────────────────────
def get_or_load_embedder(sector_id: str) -> JourneyEmbedder | None:
    """저장된 모델이 있으면 로드, 없으면 None 반환."""
    emb = JourneyEmbedder(sector_id)
    if emb.is_available():
        try:
            return emb.load()
        except Exception as e:
            logger.warning(f"임베더 로드 실패: {e}")
    return None


def run_embedding_pipeline(
    sector_id: str,
    processed_dates: list[str],
    n_clusters: int = N_CLUSTERS,
) -> JourneyEmbedder:
    """
    Dev 환경용 전체 임베딩 파이프라인.

    학습 → 저장까지 원스텝 실행.
    """
    emb = JourneyEmbedder(sector_id)
    emb.train_from_parquets(processed_dates, n_clusters=n_clusters)
    emb.save()
    return emb
