"""Read-only checks for the relocated lab03 material collection.

Checks the reviewed current inventory, every pre-move file (using preserved
originals for documented edits), formal DOCX/PDF consistency, native embedded
figures/tables, the offline gallery, and two optional-extension examples.
No model calls, training, benchmark runs, document export, or manifest writes.
"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import re
import sys
import unicodedata
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OPS = BASE / '_maintenance/relocation_20261114'
MANIFEST = OPS / 'current_manifest.json'
OLD_ROOT = 'deliverables/lab3_备份'


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        h = sha256()
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
        return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def within(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    require(path.is_relative_to(base.resolve()), 'Path outside allowed directory: ' + relative)
    return path


def historical_path(relative: str) -> Path:
    """Resolve old fingerprints to originals without rewriting historical seals."""
    rel = str(relative).replace('\\', '/')
    if rel == OLD_ROOT or rel.startswith(OLD_ROOT + '/'):
        suffix = rel[len(OLD_ROOT):].lstrip('/')
        original = within(OPS / 'before_files', suffix)
        return original if original.is_file() else within(BASE, suffix)
    if rel.startswith('deliverables/lab3/') and rel.endswith(('.docx', '.pdf')):
        original = OPS / 'before' / Path(rel).name
        if original.is_file():
            return original
    if rel == 'README.md':
        return OPS / 'before/README.md'
    return within(ROOT, rel)


def check_inventory() -> dict:
    record = load(MANIFEST)
    before = load(OPS / 'before_inventory.json')
    require(not (ROOT / OLD_ROOT).exists(), 'Old materials directory reappeared')
    errors = []
    for rel, expected in record['material_files'].items():
        path = within(BASE, rel)
        if not path.is_file() or digest(path) != expected:
            errors.append('materials/' + rel)
    for rel, expected in record['formal_files'].items():
        path = within(ROOT, rel)
        if not path.is_file() or digest(path) != expected:
            errors.append(rel)
    require(not errors, 'Current files changed since relocation review: ' + ', '.join(errors))
    originals = []
    for rel, metadata in before['files'].items():
        path = historical_path(OLD_ROOT + '/' + rel)
        if not path.is_file() or digest(path) != metadata['sha256']:
            originals.append(rel)
    require(not originals, 'Pre-move original missing or changed: ' + ', '.join(originals))
    for rel, expected in before['formal_before'].items():
        if rel.endswith(('.docx', '.pdf')):
            require(digest(historical_path(rel)) == expected, 'Pre-edit report snapshot changed: ' + rel)
    require(digest(OPS/'before/README.md') == before['readme_before'], 'Pre-edit README changed')
    return {'current_material_files': len(record['material_files']),
            'preserved_original_files': len(before['files']),
            'formal_files': len(record['formal_files'])}


def normalize(text: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).replace('\u200b', '').replace('\ufeff', '')


def check_report(folder: Path | None = None) -> dict:
    from docx import Document
    import fitz

    folder = folder or ROOT / 'deliverables/lab3'
    src = folder / '第31组_实验三_实验报告.docx'
    pdf_path = src.with_suffix('.pdf')
    old_path = OPS / 'before' / src.name
    before = Document(old_path)
    current = Document(src)
    update = load(OPS / 'report_update.json')['paragraph_updates']
    require(len(before.paragraphs) == len(current.paragraphs), 'Paragraph count changed')
    for i, (a, b) in enumerate(zip(before.paragraphs, current.paragraphs)):
        expected = update.get(str(i), {}).get('after', a.text)
        require(b.text == expected, f'Unreviewed paragraph change: {i}')
        if str(i) not in update:
            require(a._p.xml == b._p.xml, f'Unreviewed paragraph formatting: {i}')
    require(len(current.tables) == len(before.tables) == 30, 'Report table count changed')
    require(all(a._tbl.xml == b._tbl.xml for a, b in zip(before.tables, current.tables)), 'Report table changed')
    require(len(current.inline_shapes) == len(before.inline_shapes) == 14, 'Embedded figure count changed')
    with ZipFile(old_path) as a, ZipFile(src) as b:
        require(a.namelist() == b.namelist(), 'DOCX package entry set changed')
        require([n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml'],
                'Unexpected DOCX package edits')
        for name in b.namelist():
            if name.endswith('.rels'):
                require(b'TargetMode="External"' not in b.read(name), 'Report external dependency: ' + name)
    content = [p.text for p in current.paragraphs if p.text.strip()]
    content.extend(c.text for t in current.tables for r in t.rows for c in r.cells if c.text.strip())
    require(not any('lab3_备份' in text for text in content), 'Old directory remains in formal report')
    with fitz.open(pdf_path) as pdf:
        require(len(pdf) == 30, 'Report must remain 30 pages')
        text = normalize(''.join(page.get_text() for page in pdf))
        missing = [s for s in content if normalize(s) not in text]
        require(not missing, 'DOCX text absent from PDF: ' + repr(missing[:3]))
        require('lab3_备份' not in text, 'Old directory remains in PDF')
        require('outputs/lab03_materials' in text, 'New directory missing from PDF')
        for i, page in enumerate(pdf, 1):
            for word in page.get_text('words'):
                x0, y0, x1, y1 = word[:4]
                require(x0 >= 0 and y0 >= 0 and x1 <= page.rect.width + 1 and y1 <= page.rect.height + 1,
                        f'Text outside PDF page {i}: {word[4]}')
    return {'pages': 30, 'unchanged_tables': 30, 'unchanged_embedded_figures': 14,
            'matched_text_items': len(content), 'reviewed_paragraph_updates': len(update)}


class LocalReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.ids = set()

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        self.references.extend(value for key, value in attributes if key in ('src', 'href') and value)


def check_gallery() -> dict:
    parser = LocalReferences()
    parser.feed((BASE/'图片与案例索引.html').read_text(encoding='utf-8'))
    files = set()
    for ref in parser.references:
        uri = urlsplit(ref)
        if uri.scheme or uri.netloc:
            continue
        if uri.path:
            path = within(BASE, unquote(uri.path))
            require(path.is_file(), 'Gallery target missing: ' + ref)
            files.add(path)
        elif uri.fragment:
            require(unquote(uri.fragment) in parser.ids, 'Gallery anchor missing: ' + ref)
    return {'references': len(parser.references), 'local_files': len(files)}


def check_examples() -> dict:
    path = BASE/'可选拓展_20261113/verify_examples.py'
    spec = importlib.util.spec_from_file_location('lab03_material_examples', path)
    require(spec is not None and spec.loader is not None, 'Cannot load example checker')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module.run_checks()
    require(result == load(path.with_name('example_verification.json')), 'Example replay differs from frozen record')
    require(result['positive_examples_accepted'] == 2 and result['negative_cases_rejected'] == 8,
            'Example acceptance/rejection counts differ')
    return {'positive_examples': 2, 'negative_checks': 8, 'llm_calls': result['llm_calls']}


def main() -> None:
    result = {'status': 'passed', 'checked_at': datetime.now().astimezone().isoformat(),
              'inventory': check_inventory(), 'report': check_report(),
              'gallery': check_gallery(), 'examples': check_examples(),
              'scope': 'Read-only material/version checks and two given example replays; no benchmark, training, model calls, Office export, or manifest writes.'}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
