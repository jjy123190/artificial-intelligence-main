import re
from pathlib import Path

path = Path(__file__).with_name("report-source.md")
text = path.read_text(encoding="utf-8")
body, tail = text.split("# 参考文献\n", 1)
references, appendix = tail.split("# 附录A", 1)
order = list(dict.fromkeys(re.findall(r"\[(\d+)\]", body)))
assert len(order) == 8
mapping = {old: str(index) for index, old in enumerate(order, 1)}
body = re.sub(r"\[(\d+)\]", lambda match: "[" + mapping[match[1]] + "]", body)
entries = {}
for paragraph in references.strip().split("\n\n"):
    match = re.match(r"\[(\d+)\]\s*(.*)", paragraph, flags=re.S)
    if match:
        entries[match[1]] = match[2]
updated = "\n\n".join("[" + mapping[old] + "] " + entries[old] for old in order)
path.write_text(body + "# 参考文献\n\n" + updated + "\n\n# 附录A" + appendix, encoding="utf-8")
print("References follow order of first citation:", order)
