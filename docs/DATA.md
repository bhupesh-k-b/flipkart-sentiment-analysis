# Data provenance

Publisher: **Nirali Vaghani and collaborator**. Original source: [Nirali Vaghani and collaborator](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset). Retrieved 2026-10-15. Exact filenames, file sizes, download URLs, and SHA-256 checksums are recorded in [sources.json](../data/sources.json).

The source metadata reports ODbL / Database Contents License. Derived review-hash databases and analysis outputs retain the applicable source terms; original code is separate. No raw review bodies are redistributed. Publisher attribution: Nirali Vaghani and collaborator. Changes: text normalization/deduplication, stratified sampling, classification, and hash-only diagnostic exports. ODbL: https://opendatacommons.org/licenses/odbl/1-0/ ; DbCL: https://opendatacommons.org/licenses/dbcl/1-0/ .

The MIT license applies to original project code only. It does not relicense third-party data or derived data. Raw files are downloaded locally and excluded from Git.

Run `python -m src.download` from the repository root. It verifies existing files, verifies downloads before replacing them, and rejects upstream changes instead of silently analyzing a different version. Kaggle may redirect downloads to signed storage URLs; this build required no credentials. If anonymous access changes, manually download the source file to `data/raw/` and rerun checksum verification.
