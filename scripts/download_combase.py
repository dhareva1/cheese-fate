"""Download ComBase Browser search results as Excel exports.

Replays the ComBase Browser web form: runs a search, pages through the results to collect
record IDs, then requests the site's "Export to Excel" in chunks of 20 records (the site's
per-export cap). Requests are throttled and the run can be resumed: collected IDs are cached
in ``record_ids.txt`` and existing export files are skipped.

Example:
    python scripts/download_combase.py --food cheese --out data/raw/combase_cheese/exports
"""

from __future__ import annotations

import argparse
import re
import time
from html import unescape
from pathlib import Path

import requests

BASE = "https://combase.ars.usda.gov/"
P = "ctl00$ContentPlaceHolder1$"
CHUNK = 20


def hidden_fields(html: str) -> dict[str, str]:
    pattern = r'<input type="hidden" name="([^"]+)" id="[^"]*" value="([^"]*)"'
    return {unescape(m.group(1)): unescape(m.group(2)) for m in re.finditer(pattern, html)}


def record_ids(html: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"data-recordid='(\d+)'", html)))


class ComBaseSession:
    def __init__(self, food: str, response_type: str, delay: float) -> None:
        self.food = food
        self.response_type = response_type
        self.delay = delay
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (academic research data download)"
        self.page_html = ""

    def _post(self, data: dict[str, str]) -> requests.Response:
        time.sleep(self.delay)
        for attempt in range(4):
            try:
                response = self.session.post(BASE + "SearchResults.aspx", data=data, timeout=180)
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                if attempt == 3:
                    raise
                print(f"  retry after error: {error}")
                time.sleep(10 * (attempt + 1))
        raise AssertionError("unreachable")

    def search(self) -> int:
        form = hidden_fields(self.session.get(BASE + "Search.aspx", timeout=60).text)
        form.update(
            {
                P + "HiddenFood": self.food,
                P + "HiddenTypeOfResponse": self.response_type,
                P + "HiddenIsPublic": "true",
                P + "HiddenOrCondition": "1",
                P + "HiddenOrProperty": "1",
                P + "HiddenComBaseIDExactMatch": "1",
                P + "HiddenStoreLog": "no",
                P + "btnDoSearch": "Search",
            }
        )
        self.page_html = self._post(form).text
        total = re.search(r"(\d+) records", self.page_html)
        return int(total.group(1)) if total else 0

    def total_pages(self) -> int:
        return int(re.search(r'id="HiddenTotalPages" value="(\d+)"', self.page_html).group(1))

    def goto_page(self, page: int) -> list[str]:
        form = hidden_fields(self.page_html)
        form.update(
            {"__EVENTTARGET": P + "LnkPage", "__EVENTARGUMENT": "", P + "HiddenCurrentPage": str(page)}
        )
        self.page_html = self._post(form).text
        return record_ids(self.page_html)

    def export(self, ids: list[str]) -> bytes:
        form = hidden_fields(self.page_html)
        form.update({P + "HiddenExportRecords": ",".join(ids), P + "cbBtnExportToExcel": "Export"})
        response = self._post(form)
        if "spreadsheetml" not in response.headers.get("Content-Type", ""):
            raise RuntimeError(f"export did not return a workbook for IDs {ids[0]}..{ids[-1]}")
        return response.content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--food", required=True, help="ComBase food category ID, e.g. cheese")
    parser.add_argument("--type", default="any", choices=["any", "static", "dynamic"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cb = ComBaseSession(args.food, args.type, args.delay)
    total = cb.search()
    print(f"search: food={args.food} type={args.type} -> {total} records")

    id_file = args.out / "record_ids.txt"
    ids = id_file.read_text().split() if id_file.exists() else []
    if len(ids) != total:
        ids = record_ids(cb.page_html)
        pages = cb.total_pages()
        for page in range(2, pages + 1):
            ids.extend(i for i in cb.goto_page(page) if i not in ids)
            if page % 25 == 0:
                print(f"  page {page}/{pages}: {len(ids)} IDs")
        id_file.write_text("\n".join(ids) + "\n")
    print(f"collected {len(ids)} unique record IDs (expected {total})")

    chunks = [ids[i : i + CHUNK] for i in range(0, len(ids), CHUNK)]
    for n, chunk in enumerate(chunks, start=1):
        path = args.out / f"export_{n:03d}.xlsx"
        if path.exists():
            continue
        path.write_bytes(cb.export(chunk))
        print(f"  {path.name}: {len(chunk)} records")
    print("done")


if __name__ == "__main__":
    main()
