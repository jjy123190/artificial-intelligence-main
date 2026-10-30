"""Read-only source backup; verify all rows before old directories can be removed."""
import json
import sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'人工智能作业二'/'.state'/'governance.sqlite3'
target=ROOT/'.state'/'governance.sqlite3'
target.parent.mkdir(exist_ok=True)
a=sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)
b=sqlite3.connect(target)
try:
    a.backup(b)
    tables=[row[0] for row in a.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    before={table:a.execute('SELECT * FROM "'+table+'" ORDER BY 1,2').fetchall() for table in tables}
    after={table:b.execute('SELECT * FROM "'+table+'" ORDER BY 1,2').fetchall() for table in tables}
    assert before==after
    assert b.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    (ROOT/'outputs'/'qa'/'state-migration.json').write_text(json.dumps(dict(tables=tables,row_counts={table:len(rows) for table,rows in before.items()},same_rows=True,integrity='ok'),ensure_ascii=False,indent=2),encoding='utf8')
    print('Persistent state rows preserved:',{table:len(rows) for table,rows in before.items()})
finally:
    a.close();b.close()
