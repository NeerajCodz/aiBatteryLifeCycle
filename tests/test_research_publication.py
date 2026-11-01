import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reviewer4_publication_has_all_figures_and_real_downloads():
    folder = ROOT / 'frontend/public/research/reviewer4'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['revision'] == 'reviewer4-2026-11-01'
    assert [f['number'] for f in manifest['figures']] == list(range(1, 28))
    assert len(manifest['tables']) == 9
    assert len(manifest['documents']) == 3
    for asset in manifest['assets']:
        file = folder / asset['path']
        assert file.is_file()
        assert hashlib.sha256(file.read_bytes()).hexdigest() == asset['sha256']
    text = (ROOT / 'frontend/public/research_paper.md').read_text(encoding='utf-8')
    assert 'State of Health Estimation and Target Leakage Auditing' in text
    assert 'best-epoch' in text
    for n in range(1, 28):
        assert f'![Figure {n}]' in text
    assert '3.536' in text and '3.682' in text
    assert len(re.findall(r'^\|[^\n]+\|\n\| ---', text, re.MULTILINE)) == 9


def test_publication_keeps_audit_and_primary_benchmark_provenance_separate():
    folder = ROOT / 'frontend/public/research/reviewer4'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['primary_regression_artifacts_replaced'] is False
    assert manifest['source_manuscript_sha256']
    assert manifest['metric_definitions']['battery_macro_mae'] != manifest['metric_definitions']['sample_mae']
    assert 'original' in manifest['limitations'].lower()
