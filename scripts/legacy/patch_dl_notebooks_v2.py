"""
Patch notebooks 04-09 for v2:
 - Replace battery-grouped split with intra-battery chronological split
 - Update import lines to include get_version_paths
 - Update artifact save/load paths to use v2 directories
"""
import json
import re
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"

# ──────────────────────────── helpers ────────────────────────────

def load_nb(name):
    with open(NB_DIR / name, encoding="utf-8") as f:
        return json.load(f)

def save_nb(nb, name):
    with open(NB_DIR / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  ✓ Saved {name}")

def get_code_cells(nb):
    return [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]

def set_source(cell, new_src):
    """Replace cell source, splitting into line-per-element list."""
    lines = new_src.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

def src(cell):
    return "".join(cell["source"])

# ──────────── v2 intra-battery split (sequence version) ────────────
V2_SPLIT_SEQ = """\
# ── v2: intra-battery chronological split ──
# For each battery, first 80% of sequences → train, last 20% → test
train_idx, test_idx = [], []
for bid in np.unique(bids):
    idxs = np.where(bids == bid)[0]
    n = len(idxs)
    cut = int(0.8 * n)
    train_idx.extend(idxs[:cut].tolist())
    test_idx.extend(idxs[cut:].tolist())

train_idx = np.array(train_idx)
test_idx = np.array(test_idx)

X_train, y_train = X_multi[train_idx], y_multi[train_idx]
X_test, y_test = X_multi[test_idx], y_multi[test_idx]
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Batteries in both: {len(np.unique(bids))}")"""

# ──────────────────────────── NB 04 ────────────────────────────

def patch_04():
    print("Patching 04_lstm_rnn.ipynb ...")
    nb = load_nb("04_lstm_rnn.ipynb")
    cc = get_code_cells(nb)

    # Cell 0: imports — add get_version_paths, ensure_version_dirs
    s = src(cc[0][1])
    s = s.replace(
        "from src.utils.config import (\n    ARTIFACTS_DIR, FIGURES_DIR, MODELS_DIR,",
        "from src.utils.config import (\n    ARTIFACTS_DIR, FIGURES_DIR, MODELS_DIR,\n    get_version_paths, ensure_version_dirs,"
    )
    # Add v2 paths setup at end
    s += "\n\n# v2 paths\nv2 = get_version_paths('v2')\nensure_version_dirs('v2')"
    set_source(cc[0][1], s)

    # Cell 1: replace battery-grouped split with intra-battery chrono split
    s = src(cc[1][1])
    # Replace everything between "# Battery-grouped split" and the scaler code
    old_split = """# Battery-grouped split
unique_bids = np.unique(bids)
rng = np.random.RandomState(42)
rng.shuffle(unique_bids)
n_train = int(0.8 * len(unique_bids))
train_bats = set(unique_bids[:n_train])
test_bats = set(unique_bids[n_train:])

train_mask = np.isin(bids, list(train_bats))
test_mask = np.isin(bids, list(test_bats))

X_train, y_train = X_multi[train_mask], y_multi[train_mask]
X_test, y_test = X_multi[test_mask], y_multi[test_mask]
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")"""
    s = s.replace(old_split, V2_SPLIT_SEQ)
    set_source(cc[1][1], s)

    # Cell 2: model save path
    s = src(cc[2][1])
    s = s.replace(
        'MODELS_DIR / "deep" / f"{name.lower().replace(\' \', \'_\')}.pt"',
        'v2["models_deep"] / f"{name.lower().replace(\' \', \'_\')}.pt"'
    )
    set_source(cc[2][1], s)

    # Cells 3,4,5: save_fig — add v2 figures dir via FIGURES_DIR override
    for idx in [3, 4, 5]:
        s = src(cc[idx][1])
        s = s.replace('save_fig(fig, "', 'save_fig(fig, "v2_')
        set_source(cc[idx][1], s)

    # Cell 6: results save path
    s = src(cc[6][1])
    s = s.replace(
        'ARTIFACTS_DIR / "lstm_soh_results.csv"',
        'v2["results"] / "v2_lstm_soh_results.csv"'
    )
    s = s.replace(
        'Saved to artifacts/lstm_soh_results.csv',
        'Saved to v2 results'
    )
    set_source(cc[6][1], s)

    save_nb(nb, "04_lstm_rnn.ipynb")


# ──────────────────────────── NB 05 ────────────────────────────

def patch_05():
    print("Patching 05_transformer.ipynb ...")
    nb = load_nb("05_transformer.ipynb")
    cc = get_code_cells(nb)

    # Cell 0: imports
    s = src(cc[0][1])
    s = s.replace(
        "from src.utils.config import (\n    ARTIFACTS_DIR, FIGURES_DIR, MODELS_DIR,",
        "from src.utils.config import (\n    ARTIFACTS_DIR, FIGURES_DIR, MODELS_DIR,\n    get_version_paths, ensure_version_dirs,"
    )
    s += "\n\n# v2 paths\nv2 = get_version_paths('v2')\nensure_version_dirs('v2')"
    set_source(cc[0][1], s)

    # Cell 1: data + split
    s = src(cc[1][1])
    old_split = """# Battery-grouped split
unique_bids = np.unique(bids)
rng = np.random.RandomState(42)
rng.shuffle(unique_bids)
n_train = int(0.8 * len(unique_bids))
train_bats = set(unique_bids[:n_train])
test_bats = set(unique_bids[n_train:])

train_mask = np.isin(bids, list(train_bats))
test_mask = np.isin(bids, list(test_bats))

X_train, y_train = X_multi[train_mask], y_multi[train_mask]
X_test, y_test = X_multi[test_mask], y_multi[test_mask]
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")"""
    s = s.replace(old_split, V2_SPLIT_SEQ)
    set_source(cc[1][1], s)

    # Cell 2: BatteryGPT save
    s = src(cc[2][1])
    s = s.replace('MODELS_DIR / "deep" / "batterygpt.pt"', 'v2["models_deep"] / "batterygpt.pt"')
    set_source(cc[2][1], s)

    # Cell 3: TFT save
    s = src(cc[3][1])
    s = s.replace('MODELS_DIR / "deep" / "tft.pt"', 'v2["models_deep"] / "tft.pt"')
    set_source(cc[3][1], s)

    # Cell 4: iTransformer save
    s = src(cc[4][1])
    s = s.replace('MODELS_DIR / "deep" / "itransformer.keras"', 'v2["models_deep"] / "itransformer.keras"')
    set_source(cc[4][1], s)

    # Cell 5: Physics iTransformer save
    s = src(cc[5][1])
    s = s.replace('MODELS_DIR / "deep" / "physics_itransformer.keras"', 'v2["models_deep"] / "physics_itransformer.keras"')
    set_source(cc[5][1], s)

    # Cell 6: save_fig calls
    s = src(cc[6][1])
    s = s.replace('save_fig(fig, "', 'save_fig(fig, "v2_')
    set_source(cc[6][1], s)

    # Cell 7: results save
    s = src(cc[7][1])
    s = s.replace('ARTIFACTS_DIR / "transformer_soh_results.csv"', 'v2["results"] / "v2_transformer_soh_results.csv"')
    set_source(cc[7][1], s)

    save_nb(nb, "05_transformer.ipynb")


# ──────────────────────────── NB 06 ────────────────────────────

def patch_06():
    print("Patching 06_dynamic_graph.ipynb ...")
    nb = load_nb("06_dynamic_graph.ipynb")
    cc = get_code_cells(nb)

    # Cell 0: imports
    s = src(cc[0][1])
    s = s.replace(
        "from src.utils.config import (\n    ARTIFACTS_DIR, FIGURES_DIR, MODELS_DIR,",
        "from src.utils.config import (\n    ARTIFACTS_DIR, FIGURES_DIR, MODELS_DIR,\n    get_version_paths, ensure_version_dirs,"
    )
    s += "\n\n# v2 paths\nv2 = get_version_paths('v2')\nensure_version_dirs('v2')"
    set_source(cc[0][1], s)

    # Cell 1: data + split
    s = src(cc[1][1])
    old_split = """# Battery-grouped split
unique_bids = np.unique(bids)
rng = np.random.RandomState(42)
rng.shuffle(unique_bids)
n_train = int(0.8 * len(unique_bids))
train_bats = set(unique_bids[:n_train])
test_bats = set(unique_bids[n_train:])

train_mask = np.isin(bids, list(train_bats))
test_mask = np.isin(bids, list(test_bats))

X_train, y_train = X_multi[train_mask], y_multi[train_mask]
X_test, y_test = X_multi[test_mask], y_multi[test_mask]
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")"""
    s = s.replace(old_split, V2_SPLIT_SEQ)
    set_source(cc[1][1], s)

    # Cell 4: model save
    s = src(cc[4][1])
    s = s.replace('MODELS_DIR / "deep" / "dynamic_graph_itransformer.keras"',
                  'v2["models_deep"] / "dynamic_graph_itransformer.keras"')
    set_source(cc[4][1], s)

    # Cells 5,6,7: save_fig
    for idx in [5, 6, 7]:
        s = src(cc[idx][1])
        s = s.replace('save_fig(fig, "', 'save_fig(fig, "v2_')
        set_source(cc[idx][1], s)

    # Cell 7: results json save
    s = src(cc[7][1])
    s = s.replace('ARTIFACTS_DIR / "dg_itransformer_results.json"',
                  'v2["results"] / "v2_dg_itransformer_results.json"')
    set_source(cc[7][1], s)

    save_nb(nb, "06_dynamic_graph.ipynb")


# ──────────────────────────── NB 07 ────────────────────────────

def patch_07():
    print("Patching 07_vae_lstm.ipynb ...")
    nb = load_nb("07_vae_lstm.ipynb")
    cc = get_code_cells(nb)

    # Cell 0: imports
    s = src(cc[0][1])
    s = s.replace(
        "from src.utils.config import (\n    ARTIFACTS_DIR, FIGURES_DIR, MODELS_DIR,",
        "from src.utils.config import (\n    ARTIFACTS_DIR, FIGURES_DIR, MODELS_DIR,\n    get_version_paths, ensure_version_dirs,"
    )
    s += "\n\n# v2 paths\nv2 = get_version_paths('v2')\nensure_version_dirs('v2')"
    set_source(cc[0][1], s)

    # Cell 1: data + split
    s = src(cc[1][1])
    old_split = """# Battery-grouped split
unique_bids = np.unique(bids)
rng = np.random.RandomState(42)
rng.shuffle(unique_bids)
n_train = int(0.8 * len(unique_bids))
train_bats = set(unique_bids[:n_train])
test_bats = set(unique_bids[n_train:])

train_mask = np.isin(bids, list(train_bats))
test_mask = np.isin(bids, list(test_bats))

X_train, y_train = X_multi[train_mask], y_multi[train_mask]
X_test, y_test = X_multi[test_mask], y_multi[test_mask]
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")"""
    s = s.replace(old_split, V2_SPLIT_SEQ)
    set_source(cc[1][1], s)

    # Cell 2: model save
    s = src(cc[2][1])
    s = s.replace('MODELS_DIR / "deep" / "vae_lstm.pt"', 'v2["models_deep"] / "vae_lstm.pt"')
    set_source(cc[2][1], s)

    # Cells 4,5,6: save_fig
    for idx in [4, 5, 6]:
        s = src(cc[idx][1])
        s = s.replace('save_fig(fig, "', 'save_fig(fig, "v2_')
        set_source(cc[idx][1], s)

    # Cell 6: results json save
    s = src(cc[6][1])
    s = s.replace('ARTIFACTS_DIR / "vae_lstm_results.json"',
                  'v2["results"] / "v2_vae_lstm_results.json"')
    set_source(cc[6][1], s)

    save_nb(nb, "07_vae_lstm.ipynb")


# ──────────────────────────── NB 08 ────────────────────────────

def patch_08():
    print("Patching 08_ensemble.ipynb ...")
    nb = load_nb("08_ensemble.ipynb")
    cc = get_code_cells(nb)

    # Cell 0: imports
    s = src(cc[0][1])
    s = s.replace(
        "from src.utils.config import ARTIFACTS_DIR, FIGURES_DIR, MODELS_DIR",
        "from src.utils.config import ARTIFACTS_DIR, FIGURES_DIR, MODELS_DIR, get_version_paths, ensure_version_dirs"
    )
    s += "\n\n# v2 paths\nv2 = get_version_paths('v2')\nensure_version_dirs('v2')"
    set_source(cc[0][1], s)

    # Cell 1: data + split
    s = src(cc[1][1])
    old_split = """# Battery-grouped split
unique_bids = np.unique(bids)
rng = np.random.RandomState(42)
rng.shuffle(unique_bids)
n_train = int(0.8 * len(unique_bids))
train_bats = set(unique_bids[:n_train])
test_bats = set(unique_bids[n_train:])

train_mask = np.isin(bids, list(train_bats))
test_mask = np.isin(bids, list(test_bats))

X_train, y_train = X_multi[train_mask], y_multi[train_mask]
X_test, y_test = X_multi[test_mask], y_multi[test_mask]
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")"""
    s = s.replace(old_split, V2_SPLIT_SEQ)
    set_source(cc[1][1], s)

    # Cell 2: model loading paths — load from v2 deep models
    s = src(cc[2][1])
    s = s.replace('MODELS_DIR / "deep" / f"{name}.pt"', 'v2["models_deep"] / f"{name}.pt"')
    set_source(cc[2][1], s)

    # Cells 4,5,6: save_fig
    for idx in [4, 5, 6]:
        s = src(cc[idx][1])
        s = s.replace('save_fig(fig, "', 'save_fig(fig, "v2_')
        set_source(cc[idx][1], s)

    # Cell 5: results save
    s = src(cc[5][1])
    s = s.replace('ARTIFACTS_DIR / "ensemble_results.csv"',
                  'v2["results"] / "v2_ensemble_results.csv"')
    set_source(cc[5][1], s)

    save_nb(nb, "08_ensemble.ipynb")


# ──────────────────────────── NB 09 ────────────────────────────

def patch_09():
    print("Patching 09_evaluation.ipynb ...")
    nb = load_nb("09_evaluation.ipynb")
    cc = get_code_cells(nb)

    # Cell 0: imports
    s = src(cc[0][1])
    s = s.replace(
        "from src.utils.config import ARTIFACTS_DIR, FIGURES_DIR",
        "from src.utils.config import ARTIFACTS_DIR, FIGURES_DIR, get_version_paths"
    )
    s += "\n\n# v2 paths\nv2 = get_version_paths('v2')"
    set_source(cc[0][1], s)

    # Cell 1: result loading paths → v2
    s = src(cc[1][1])
    s = s.replace('ARTIFACTS_DIR / "classical_soh_results.csv"',
                  'v2["results"] / "v2_classical_soh_results.csv"')
    s = s.replace('ARTIFACTS_DIR / "lstm_soh_results.csv"',
                  'v2["results"] / "v2_lstm_soh_results.csv"')
    s = s.replace('ARTIFACTS_DIR / "transformer_soh_results.csv"',
                  'v2["results"] / "v2_transformer_soh_results.csv"')
    s = s.replace('ARTIFACTS_DIR / "dg_itransformer_results.json"',
                  'v2["results"] / "v2_dg_itransformer_results.json"')
    s = s.replace('ARTIFACTS_DIR / "vae_lstm_results.json"',
                  'v2["results"] / "v2_vae_lstm_results.json"')
    s = s.replace('ARTIFACTS_DIR / "ensemble_results.csv"',
                  'v2["results"] / "v2_ensemble_results.csv"')
    set_source(cc[1][1], s)

    # Cell 2: unified results save
    s = src(cc[2][1])
    s = s.replace('ARTIFACTS_DIR / "unified_results.csv"',
                  'v2["results"] / "v2_unified_results.csv"')
    set_source(cc[2][1], s)

    # Cell 3: save_fig
    s = src(cc[3][1])
    s = s.replace('save_fig(fig, "', 'save_fig(fig, "v2_')
    set_source(cc[3][1], s)

    # Cell 5: CED split — replace battery-grouped with chrono split
    s = src(cc[5][1])
    old_split_09 = """unique_bids = np.unique(bids)
rng = np.random.RandomState(42)
rng.shuffle(unique_bids)
n_train = int(0.8 * len(unique_bids))
train_bats = set(unique_bids[:n_train])
test_mask = ~np.isin(bids, list(train_bats))
y_test = y_all[test_mask]
bids_test = bids[test_mask]"""
    new_split_09 = """# v2: intra-battery chronological split for CED
test_idx = []
for bid in np.unique(bids):
    idxs = np.where(bids == bid)[0]
    cut = int(0.8 * len(idxs))
    test_idx.extend(idxs[cut:].tolist())
test_idx = np.array(test_idx)
y_test = y_all[test_idx]
bids_test = bids[test_idx]"""
    s = s.replace(old_split_09, new_split_09)
    # Also update save_fig if present
    s = s.replace('save_fig(fig, "', 'save_fig(fig, "v2_')
    set_source(cc[5][1], s)

    # Cell 7: final rankings save
    s = src(cc[7][1])
    s = s.replace('ARTIFACTS_DIR / "final_rankings.csv"',
                  'v2["results"] / "v2_final_rankings.csv"')
    set_source(cc[7][1], s)

    save_nb(nb, "09_evaluation.ipynb")


# ──────────────────────────── main ────────────────────────────

if __name__ == "__main__":
    patch_04()
    patch_05()
    patch_06()
    patch_07()
    patch_08()
    patch_09()
    print("\nAll 6 notebooks patched for v2!")
