# backend/io_utils.py
import io, json, pandas as pd

TABULAR_EXT = {"csv","tsv","txt","parquet","json","xlsx","xls"}

def _read_guess_sep(b):  # for csv/tsv/txt
    try:
        sample = b.decode("utf-8", errors="ignore")[:2000]
        if "\t" in sample: return "\t"
        if ";" in sample: return ";"
        return ","
    except Exception:
        return ","

def read_any_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(file_bytes))
    if name.endswith(".json"):
        obj = json.loads(file_bytes.decode("utf-8", "ignore"))
        # allow list of records or dict of columns
        return pd.DataFrame(obj)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    if name.endswith(".csv") or name.endswith(".tsv") or name.endswith(".txt"):
        sep = _read_guess_sep(file_bytes)
        return pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    # fallback: try pandas sniffer in order
    for fn in (pd.read_parquet, pd.read_json, pd.read_excel, pd.read_csv):
        try:
            return fn(io.BytesIO(file_bytes))
        except Exception:
            continue
    raise ValueError(f"Unsupported or unreadable file: {filename}")
