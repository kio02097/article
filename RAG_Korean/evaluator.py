import os
import ast
import unicodedata
import re
import numpy as np
import pandas as pd
from langchain_core.embeddings import Embeddings
from config import LLM_MODEL, PROMPT_PATH, RESULTS_DIR, SUMMARY_DIR
from utils import load_prompt_file

class RAGEvaluator:
    def __init__(
        self,
        dataset_csv: str,
        collection_name: str,
        k: int,
        embedding_function: Embeddings,
        chroma_dir: str,
        llm_model: str = LLM_MODEL,
        prompt_path: str = PROMPT_PATH,
        do_llm: bool = True,
        save_contexts: bool = True,
    ):
        self.dataset_csv = dataset_csv
        self.collection_name = collection_name
        self.k = k
        self.embedding_function = embedding_function
        self.chroma_dir = chroma_dir
        self.llm_model = llm_model
        self.prompt_path = prompt_path
        self.do_llm = do_llm
        self.save_contexts = save_contexts

        dataset_name = os.path.splitext(os.path.basename(dataset_csv))[0]
        self.run_name = f"{dataset_name}__{self.collection_name}_k{self.k}"

        self.result_csv_path = os.path.join(RESULTS_DIR, self.run_name + ".csv")
        self.eval_csv_path = os.path.join(SUMMARY_DIR, self.run_name + "_retrieval.csv")
        self.debug_csv_path = os.path.join(SUMMARY_DIR, self.run_name + "_debug_retrieval.csv")

        self.df_questions = None
        self.df_results = None
        self.df_eval = None

        self._init_vectorstore()
        if self.do_llm:
            self._init_llm_and_prompt()

    def _init_vectorstore(self):
        # 노트북에 명시된 Chroma 벡터스토어 연동 로직
        from langchain_chroma import Chroma
        self.vectorstore = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embedding_function,
            collection_name=self.collection_name
        )

    def _init_llm_and_prompt(self):
        self.sys_prompt, self.user_prompt = load_prompt_file(self.prompt_path)
        print("LLM 및 Prompt 초기화 완료.")

    @staticmethod
    def _maybe_parse(x):
        if isinstance(x, str):
            s = x.strip()
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                try: return ast.literal_eval(s)
                except Exception: return x
        return x

    @staticmethod
    def _normalize(text):
        if text is None: return ""
        return unicodedata.normalize("NFC", str(text)).strip()

    @staticmethod
    def _unwrap_singleton(x, max_depth=3):
        for _ in range(max_depth):
            if isinstance(x, list) and len(x) == 1: x = x[0]
            else: break
        return x

    @staticmethod
    def _extract_pmid(value):
        if value is None: return None
        if isinstance(value, int): return str(value)
        if isinstance(value, dict):
            if "pmid" in value: return RAGEvaluator._extract_pmid(value["pmid"])
            if "document" in value: return RAGEvaluator._extract_pmid(value["document"])
        s = str(value).strip()
        if s.isdigit(): return s
        return None


# ---------- 평가지표 독립 함수 모듈 ----------
def compute_token_f1(gold: set, pred: set) -> float:
    if not gold and not pred: return 1.0
    if not gold or not pred: return 0.0
    tp = len(gold & pred)
    prec = tp / len(pred)
    rec = tp / len(gold)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

def compute_yesno_macro_f1(df: pd.DataFrame) -> float:
    for c in ["yesno_macro_f1", "yesno_f1", "macro_f1", "MacroF1", "exact_yesno_macro_f1"]:
        if c in df.columns:
            return float(pd.to_numeric(df[c], errors="coerce").mean())
    return np.nan