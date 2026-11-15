"""对照组：关系表表示（SQLite）。用于和知识图谱对比"同一知识、不同表示"下的推理效率。"""
import sqlite3


class RelationalStore:
    def __init__(self, holdings, index=True):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE holdings(holder TEXT, company TEXT, ratio REAL)")
        self.conn.executemany("INSERT INTO holdings VALUES (?,?,?)", holdings)
        if index:
            self.conn.execute("CREATE INDEX idx_company ON holdings(company)")

    @staticmethod
    def join_sql(depth):
        """查询第 depth 层上游股东需要 depth 张表自连接 —— SQL 随层数变长。"""
        tables = [f"holdings h{i}" for i in range(1, depth + 1)]
        joins = " ".join(f"JOIN {tables[i]} ON h{i + 1}.company = h{i}.holder" for i in range(1, depth))
        ratio = " * ".join(f"h{i}.ratio" for i in range(1, depth + 1))
        return f"SELECT h{depth}.holder, {ratio} FROM {tables[0]} {joins} WHERE h1.company = ?"

    def upstream_by_joins(self, company, max_depth):
        """方式一：多表自连接，逐层查询 1..max_depth 层的所有上游股东。"""
        rows = []
        for d in range(1, max_depth + 1):
            rows += [(h, r, d) for h, r in self.conn.execute(self.join_sql(d), (company,))]
        return rows

    def upstream_recursive(self, company, max_depth):
        """方式二：递归 CTE（SQL:1999 引入，专门为了弥补关系模型做"多跳"查询的不足）。"""
        sql = """
        WITH RECURSIVE up(holder, ratio, lvl, path) AS (
            SELECT holder, ratio, 1, '|' || company || '|' || holder || '|'
            FROM holdings WHERE company = ?
            UNION ALL
            SELECT h.holder, up.ratio * h.ratio, up.lvl + 1, up.path || h.holder || '|'
            FROM holdings h JOIN up ON h.company = up.holder
            WHERE up.lvl < ? AND instr(up.path, '|' || h.holder || '|') = 0
        )
        SELECT holder, ratio, lvl FROM up
        """
        return list(self.conn.execute(sql, (company, max_depth)))
