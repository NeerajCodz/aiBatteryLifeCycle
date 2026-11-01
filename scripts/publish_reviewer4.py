"""Publish a verified manuscript as static web assets, without retraining models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
URL = '/research/reviewer4/'


def paragraph_text(element):
    return ''.join((node.text or '') if node.tag == W + 't' else '\n'
                   for node in element.iter() if node.tag in {W + 't', W + 'br'}).strip()


def publish(review_root: Path, public: Path) -> dict:
    verification = json.loads((review_root / 'reproducibility/verification_final.json').read_text())
    if not verification['scientific_checks_passed'] or not verification['document_checks_passed']:
        raise ValueError('Publication requires successful scientific and document verification')
    figures = json.loads((review_root / 'figure_manifest.json').read_text(encoding='utf-8'))
    if [f['number'] for f in figures] != list(range(1, 28)):
        raise ValueError('Expected all 27 sequential Reviewer 4 figures')
    target = public / 'research/reviewer4'
    target.mkdir(parents=True, exist_ok=True)
    assets = []

    def save(relative, data):
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        assets.append({'path': relative, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
        return URL + relative

    documents = []
    for title, source, name in [
        ('Manuscript PDF', 'final_mdp_r4.pdf', 'final_mdp_r4.pdf'),
        ('Reviewer response PDF', 'qa/response/response.pdf', 'response_to_reviewer4.pdf'),
        ('Reproducibility supplement PDF', 'qa/supplement/supplement.pdf', 'reproducibility_supplement.pdf'),
    ]:
        data = (review_root / source).read_bytes()
        if not data.startswith(b'%PDF-'):
            raise ValueError(f'Invalid PDF: {source}')
        documents.append({'title': title, 'url': save(name, data)})
    for figure in figures:
        figure['downloads'] = {}
        for extension in ['svg', 'pdf']:
            relative = f"figures/{figure['stem']}.{extension}"
            figure['downloads'][extension] = save(relative, (review_root / relative).read_bytes())
        figure['url'] = figure['downloads']['svg']
    # Exact machine-readable manuscript tables plus separately identified audit tables.
    manuscript = review_root / 'final_mdp_r4.docx'
    with zipfile.ZipFile(manuscript) as archive:
        body = ET.fromstring(archive.read('word/document.xml')).find(W + 'body')
    markdown, tables = [], []
    figure_index = 0
    elements = []
    for position, element in enumerate(body):
        # The author block and two-figure layout are Word layout tables, not
        # numbered data tables. Preserve their paragraphs and nested drawings.
        if element.tag == W + 'tbl' and (position == 1 or element.find('.//' + W + 'drawing') is not None):
            elements.extend(element.iter(W + 'p'))
        else:
            elements.append(element)
    for element in elements:
        if element.tag == W + 'p':
            if element.find('.//' + W + 'drawing') is not None:
                figure = figures[figure_index]
                markdown.append(f"![Figure {figure['number']}]({figure['url']})")
                figure_index += 1
            text = paragraph_text(element)
            if not text:
                continue
            if not markdown:
                text = '# ' + text
            elif re.match(r'^[IVX]+\.\s+[A-Z]', text):
                text = '## ' + text
            elif re.match(r'^[A-Z]\.\s+[A-Z]', text):
                text = '### ' + text
            markdown.append(text)
        elif element.tag == W + 'tbl':
            rows = [[paragraph_text(cell) for cell in row.findall(W + 'tc')]
                    for row in element.findall(W + 'tr')]
            if not rows:
                continue
            stream = io.StringIO(newline='')
            csv.writer(stream).writerows(rows)
            n = len(tables) + 1
            url = save(f'tables/table_{n:02d}.csv', stream.getvalue().encode('utf-8'))
            tables.append({'number': n, 'url': url})
            width = max(map(len, rows))
            normalized = [[cell.replace('|', '\\|').replace('\n', ' ') for cell in row] + [''] * (width-len(row)) for row in rows]
            markdown.append('\n'.join(['| ' + ' | '.join(normalized[0]) + ' |',
                             '| ' + ' | '.join(['---'] * width) + ' |'] +
                            ['| ' + ' | '.join(row) + ' |' for row in normalized[1:]]))
            markdown.append(f'[Download Table {n} as CSV]({url})')
    if figure_index != 27 or len(tables) != 9:
        raise ValueError('Manuscript figure/table completeness check failed')
    for source in sorted((review_root / 'reproducibility/tables').glob('*.csv')):
        save('audit/' + source.name, source.read_bytes())
    for name in ['verification_final.json', 'verification_scientific.json', 'verification_documents.json',
                 'reference_assessment.json', 'text_change_audit.json']:
        save('audit/' + name, (review_root / 'reproducibility' / name).read_bytes())
    manifest = {
        'revision': 'reviewer4-2026-11-01',
        'title': paragraph_text(body.find(W + 'p')),
        'source_manuscript_sha256': hashlib.sha256(manuscript.read_bytes()).hexdigest(),
        'primary_regression_artifacts_replaced': False,
        'documents': documents, 'figures': figures, 'tables': tables, 'assets': assets,
        'metric_definitions': {
            'sample_mae': 'Mean cycle error after averaging repeated held-out predictions per cycle.',
            'battery_macro_mae': 'Equal-weight mean of per-battery MAEs after cycle prediction averaging.',
            'fold_mean_mae': 'Equal-weight mean of the 15 seed-fold MAEs before prediction pooling.',
        },
        'limitations': 'Original validation histories and best-epoch identifiers were not recoverable. Secondary reruns remain separately labeled. Inference is conditional and exploratory; CALCE has four independent cells.',
    }
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    rendered = '\n\n'.join(markdown) + '\n'
    (public / 'research_paper.md').write_text(rendered, encoding='utf-8')
    (ROOT / 'docs/research_paper.md').write_text(rendered, encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, default=ROOT.parent / 'paper/revision-2026-11-r4')
    parser.add_argument('--public-dir', type=Path, default=ROOT / 'frontend/public')
    args = parser.parse_args()
    result = publish(args.review_root, args.public_dir)
    print(f"Published {len(result['figures'])} figures, {len(result['tables'])} tables and {len(result['documents'])} PDFs")
