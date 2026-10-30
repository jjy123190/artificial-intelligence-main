from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.parse import quote
import json

OUT = Path(__file__).resolve().parent
DOIS = [
    "10.1145/321250.321253",
    "10.1016/S0019-9958(65)90241-X",
    "10.1016/S0020-7373(75)80002-2",
    "10.1214/aoms/1177698950",
    "10.1016/0020-0255(87)90007-7",
    "10.1109/69.43410",
]

def fetch(doi):
    failures = []
    for url, accept, crossref in [
        ("https://api.crossref.org/works/" + quote(doi, safe=""), "application/json", True),
        ("https://doi.org/" + quote(doi, safe="/"), "application/vnd.citationstyles.csl+json", False),
    ]:
        try:
            req = Request(url, headers={"User-Agent": "CourseworkBibliographyVerifier/1.0", "Accept": accept})
            with urlopen(req, timeout=20) as response:
                data = json.load(response)
            record = data["message"] if crossref else data
            return {"doi": doi, "status": "verified_metadata", "source": url, "metadata": record}
        except Exception as exc:
            failures.append(str(exc))
    return {"doi": doi, "status": "unverified", "errors": failures}

def main():
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(fetch, DOIS))
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "records": records}
    (OUT / "references-verified.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in records:
        data = item.get("metadata", {})
        print(json.dumps({"doi": item["doi"], "status": item["status"], "title": data.get("title"),
                          "author": data.get("author"), "container": data.get("container-title"),
                          "volume": data.get("volume"), "issue": data.get("issue"), "pages": data.get("page"),
                          "year": data.get("issued"), "errors": item.get("errors")}, ensure_ascii=True), flush=True)

if __name__ == "__main__":
    main()
