# ComBase: Cheese (all organisms)

## Source

- Database: [ComBase Browser](https://combase.ars.usda.gov/Search.aspx) (USDA ARS), accessed 2026-10-04
- Search criteria: Food category = `Cheese` (ID `cheese`), response type = Any, no other filters
- Result set: 4561 records

## Contents

- `batch_01.xlsx`, `batch_02.xlsx` — earlier manual exports (20 unique records; `batch_02`
  contains all records of `batch_01`). Superseded by `exports/`.
- `exports/export_001.xlsx` … `export_229.xlsx` — complete download, 20 records per file
  (the site's export cap) except `export_229.xlsx` (1 record). 4561 rows, 4561 unique Record IDs.
- `exports/record_ids.txt` — internal ComBase record IDs in search-result order, used for the
  export requests.

Downloaded with [`scripts/download_combase.py`](../../../scripts/download_combase.py):

```bash
python scripts/download_combase.py --food cheese --out data/raw/combase_cheese/exports
```

## Workbook layout

Each workbook has a `Data Records` sheet (one row per record) and, when the records in that file
have them, long-format time-series sheets keyed by `Record ID`:

| Sheet | Columns |
| --- | --- |
| `Logcs` | `Record ID, Time, Logc` |
| `Temperatures` | `Record ID, Time, Temperature` |
| `pHs` | `Record ID, Time, pH` |
| `Aws` | `Record ID, Time, Aw` |

A sheet is omitted when no record in that file has the variable (e.g. 23 files have only
`Data Records` and `Logcs`). Concatenated sheets contain duplicate rows; drop them before use.

## Initial audit (not yet curated)

- Organisms: 15 groups. Largest: lactic acid bacteria (1352), *Listeria monocytogenes/innocua*
  (500), *E. coli* (390), aerobic total spoilage bacteria (305), Enterobacteriaceae (298),
  *S. aureus* (285).
- 4187 records have ≥3 log-count points.
- 2772 records have more than one distinct temperature value over time; 3118 have a pH range
  > 0.3. 2480 records have both varying temperature and ≥5 log-count points.
- ComBase's "dynamic" filter (3921 records) means a log-count time series exists, not that the
  environment varies; time-varying conditions must be checked from the condition sheets.
- Many records are mixed populations (e.g. "Aerobic total spoilage bacteria", "Lactic acid
  bacteria") rather than single species, and temperature profiles are sparse step series
  (cheese-making steps followed by ripening), so interpolation assumptions must be recorded.
- Log-count value `-0.01` is ComBase's symbol for "not detected" (below the detection limit),
  not a count; a few sources use `0.01` the same way. Detection limits are not recorded.
  Records without condition sheets may hold a constant `Temperature (C)`, `pH` or `Aw` in
  `Data Records`; some `Aw` values there are implausible (0.24–0.33). Handling is implemented
  in `src/cheese_fate/` and documented in the study protocol (§3).
