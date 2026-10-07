from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridIndex:
    """
    Hybrid Retrieval Engine combining Sparse (BM25) and Dense (SentenceTransformer)
    indexing using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)

    def reciprocal_rank_fusion(
        self,
        bm25_ranks: List[int],
        dense_ranks: List[int],
        k: int = 60
    ) -> List[float]:
        """
        Calculates RRF score: RRF_score(d) = (1 / (k + rank_bm25)) + (1 / (k + rank_dense))
        """
        rrf_scores = []
        for r_bm25, r_dense in zip(bm25_ranks, dense_ranks):
            score = (1.0 / (k + r_bm25)) + (1.0 / (k + r_dense))
            rrf_scores.append(score)
        return rrf_scores

    def rank_passages(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not passages:
            return []

        texts = [p.get("snippet", "") or p.get("text", "") for p in passages]
        tokenized_corpus = [t.lower().split() for t in texts]
        tokenized_query = query.lower().split()

        # 1. Sparse BM25 Scoring
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_order = np.argsort(bm25_scores)[::-1]
        bm25_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(bm25_order)}

        # 2. Dense Vector Embeddings Scoring
        query_vec = self.embedding_model.encode(query, convert_to_tensor=True)
        passage_vecs = self.embedding_model.encode(texts, convert_to_tensor=True)
        
        # Cosine Similarity
        similarity_scores = (query_vec @ passage_vecs.T).cpu().numpy()
        dense_order = np.argsort(similarity_scores)[::-1]
        dense_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(dense_order)}

        # 3. Reciprocal Rank Fusion
        bm25_rank_list = [bm25_ranks[i] for i in range(len(passages))]
        dense_rank_list = [dense_ranks[i] for i in range(len(passages))]
        rrf_scores = self.reciprocal_rank_fusion(bm25_rank_list, dense_rank_list)

        # Attach RRF scores and sort
        ranked_passages = []
        for idx, passage in enumerate(passages):
            p_copy = dict(passage)
            p_copy["rrf_score"] = float(rrf_scores[idx])
            ranked_passages.append(p_copy)

        ranked_passages.sort(key=lambda x: x["rrf_score"], reverse=True)
        return ranked_passages[:top_k]
