from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks candidate evidence passages using a deep Cross-Encoder model
    (cross-encoder/ms-marco-MiniLM-L-6-v2) for exact query-passage relevance.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.reranker = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not passages:
            return []

        pairs = [[query, p.get("snippet", "") or p.get("text", "")] for p in passages]
        scores = self.reranker.predict(pairs)

        # Normalize cross-encoder logits to 0.0 - 1.0 sigmoid probabilities
        import numpy as np
        sigmoid_scores = 1.0 / (1.0 + np.exp(-scores))

        reranked_passages = []
        for idx, passage in enumerate(passages):
            p_copy = dict(passage)
            p_copy["relevance_score"] = float(sigmoid_scores[idx])
            reranked_passages.append(p_copy)

        reranked_passages.sort(key=lambda x: x["relevance_score"], reverse=True)
        return reranked_passages[:top_k]
