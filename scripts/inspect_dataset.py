"""First thing to run on the real HHGOA_IEEE folder: shows how config/datamap.yaml resolves.

    python scripts/inspect_dataset.py [DATA_DIR]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if len(sys.argv) > 1:
    import os
    os.environ["DATA_DIR"] = sys.argv[1]

from fraud_agent import settings  # noqa: E402
from fraud_agent.data.dataset import find_files, pick, read_table, read_text  # noqa: E402

root = settings.DATA_DIR
print("DATA_DIR:", root)
for f in sorted(p for p in root.rglob("*") if p.is_file()):
    print(f"  {f.relative_to(root)}  ({f.stat().st_size / 1e6:.1f} MB)")
print()
for kind in settings.datamap()["files"]:
    print(f"[{kind}] ->", [str(p.relative_to(root)) for p in find_files(kind)])
print()
for kind, section in (("transactions", "transaction"), ("identity", "identity"), ("closed_cases", "case"), ("benchmark", "case")):
    files = [f for f in find_files(kind) if f.suffix.lower() in (".csv", ".parquet", ".json", ".jsonl", ".xlsx")]
    if not files:
        print(f"!! no {kind} file matched"); continue
    df = read_table(files[0]).head(200) if files[0].suffix != ".csv" else __import__("pandas").read_csv(files[0], nrows=200)
    print(f"== {kind}: {files[0].name}  columns={len(df.columns)}")
    print("   ", list(df.columns)[:40])
    for k, cands in settings.datamap()[section].items():
        cands = cands if isinstance(cands, list) else [cands]
        print(f"    {k:14s} -> {pick(df, cands) if k != 'card' else [pick(df, [c]) for c in cands]}")
    print(df.head(3).T.head(25).to_string())
readme = [f for f in root.rglob("*") if f.stem.lower() == "readme"]
if readme:
    print("\n==== README ====\n", read_text(readme[0])[:6000])
