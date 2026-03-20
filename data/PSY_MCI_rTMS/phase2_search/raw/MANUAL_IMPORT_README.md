# Manual Database Imports

Place exported .ris or .csv files here from databases without API access.

## Supported Formats

- `.ris` — RIS format (Cochrane, Embase, Web of Science, Ovid, etc.)
- `.csv` — CSV format (must have columns: Title, Abstract, Authors, Year, DOI)

## Naming Convention

The filename (without extension) becomes the **source label** in the PRISMA flow
and Methods section. Use descriptive names:

```
cochrane_search.ris      → Source: "cochrane_search"
embase_export.ris        → Source: "embase_export"  
wos_core_collection.ris  → Source: "wos_core_collection"
```

## How to Export

### Cochrane Library
1. Go to https://www.cochranelibrary.com/advanced-search
2. Run your search query
3. Select All → Export Selected → RIS format
4. Save as `cochrane_search.ris` in this folder

### Embase (via Ovid)
1. Log in to Ovid → Select Embase
2. Run your search strategy
3. Select results → Export → RIS format
4. Save as `embase_export.ris` in this folder

### Web of Science
1. Go to Web of Science → run search
2. Export → Other File Formats → select RIS
3. Save as `wos_export.ris` in this folder

## After Placing Files

Run deduplication to merge with API results:
```bash
python scripts/run_phase2.py --deduplicate
```

The counts from each file will appear in `search_log.json` and the PRISMA flow.
