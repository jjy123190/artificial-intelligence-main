"""本地不可覆盖的案例版本账本。写入使用SQLite事务和乐观版本检查。

原始data/raw不回写；凭证保存在明确的本地数据库，不接受任意文件路径。
历史证据与计算快照不覆盖，实时日期快照自动排除失效证据并展示撤回差异。
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
from contextlib import contextmanager
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .governance import governance_snapshot, claim_changes


class RevisionConflict(ValueError):
    pass


def _encoded(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class GovernanceHistory:
    def __init__(self, database: Path, runner=governance_snapshot):
        self.database = database.resolve()
        self.runner = runner
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS revisions (case_id TEXT NOT NULL, revision INTEGER NOT NULL, "
                               "name TEXT NOT NULL, created_utc TEXT NOT NULL, payload TEXT NOT NULL, "
                               "snapshot TEXT NOT NULL, previous_hash TEXT NOT NULL, hash TEXT NOT NULL, "
                               "PRIMARY KEY(case_id, revision))")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def list_cases(self):
        with self._connect() as connection:
            rows = connection.execute("SELECT r.case_id,r.revision,r.name,r.created_utc FROM revisions r "
                                      "JOIN (SELECT case_id,MAX(revision) revision FROM revisions GROUP BY case_id) v "
                                      "ON r.case_id=v.case_id AND r.revision=v.revision ORDER BY r.created_utc DESC").fetchall()
        return [dict(row) for row in rows]

    def _rows(self, case_id):
        if not isinstance(case_id, str) or len(case_id) != 32 or any(c not in "0123456789abcdef" for c in case_id):
            raise ValueError("案例ID格式不合法")
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM revisions WHERE case_id=? ORDER BY revision", (case_id,)).fetchall()
        if not rows:
            raise KeyError("未找到已保存案例")
        previous = ""
        for expected, row in enumerate(rows, 1):
            digest = hashlib.sha256((row["case_id"] + str(row["revision"]) + row["name"] + row["created_utc"]
                     + row["payload"] + row["snapshot"] + row["previous_hash"]).encode("utf-8")).hexdigest()
            if row["revision"] != expected or row["previous_hash"] != previous or row["hash"] != digest:
                raise ValueError("历史版本内容或哈希链不一致，拒绝使用")
            previous = row["hash"]
        return rows

    def load(self, case_id, revision=None):
        rows = self._rows(case_id)
        if revision is not None and (type(revision) is not int or revision < 1 or revision > len(rows)):
            raise ValueError("版本号不存在")
        row = rows[-1] if revision is None else rows[revision - 1]
        return {"case_id": case_id, "revision": row["revision"], "name": row["name"],
                "created_utc": row["created_utc"], "payload": json.loads(row["payload"]),
                "saved_snapshot": json.loads(row["snapshot"]), "hash": row["hash"],
                "previous_hash": row["previous_hash"]}

    def history(self, case_id):
        return [{k: row[k] for k in ("revision", "name", "created_utc", "hash", "previous_hash")}
                for row in self._rows(case_id)]

    def evaluate(self, case_id, as_of, revision=None):
        record = self.load(case_id, revision)
        payload = {**record["payload"], "as_of": as_of}
        current = self.runner(payload)
        return {**record, "payload": payload, "snapshot": current,
                "changes": claim_changes(record["saved_snapshot"], current),
                "history": self.history(case_id), "persisted": True}

    def save(self, name, payload, case_id=None, expected_revision=0):
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("案例名称须为1至120字符")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("版本号须为非负整数")
        name = name.strip()
        current = self.runner(payload)
        previous = self.load(case_id) if case_id else None
        identifier = case_id or uuid.uuid4().hex
        actual = previous["revision"] if previous else 0
        if actual != expected_revision:
            raise RevisionConflict("案例已被其他页面更新，请重新加载后再保存")
        created = datetime.now(timezone.utc).isoformat()
        revision = actual + 1
        old_hash = previous["hash"] if previous else ""
        encoded_payload, encoded_snapshot = _encoded(payload), _encoded(current)
        digest = hashlib.sha256((identifier + str(revision) + name + created + encoded_payload
                                 + encoded_snapshot + old_hash).encode("utf-8")).hexdigest()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = connection.execute("SELECT MAX(revision) FROM revisions WHERE case_id=?", (identifier,)).fetchone()[0] or 0
            if latest != expected_revision:
                raise RevisionConflict("保存发生版本冲突，请重新加载后重试")
            connection.execute("INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?)",
                (identifier, revision, name.strip(), created, encoded_payload, encoded_snapshot, old_hash, digest))
        return {"case_id": identifier, "revision": revision, "name": name.strip(), "created_utc": created,
                "payload": payload, "snapshot": current,
                "changes": claim_changes(previous["saved_snapshot"] if previous else None, current),
                "history": self.history(identifier), "hash": digest, "persisted": True}
