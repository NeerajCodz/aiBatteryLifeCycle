# Benchmark data sources

Raw archives are downloaded into `datasets/raw/` and are never committed. The
download script records SHA-256 values in `datasets/checksums.sha256` after
transfer so each experimental run can identify its exact inputs.

| Dataset | Cells used | Official source | Citation / terms |
| --- | --- | --- | --- |
| NASA PCoE Li-ion Battery Aging | 30 retained identifiers after the documented B0049–B0052 quality exclusion | [NASA Open Data record](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets), [NASA-hosted archive mirror](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip) | Saha and Goebel, *Battery Data Set*, NASA Ames PCoE, 2007 |
| CALCE CS2 | CS2_35, CS2_36, CS2_37, CS2_38 | [CALCE Battery Data Archive](https://web.calce.umd.edu/batteries/data/) | Publications must cite the archive and at least three papers listed by CALCE |
| Oxford Battery Degradation Dataset 1 | Cell1-Cell8 | [Oxford University Research Archive](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) | D. Howey and C. Birkl, *Oxford Battery Degradation Dataset 1*, University of Oxford, 2017, DOI 10.5287/bodleian:KO2kdmYGg; ODC-ODbL 1.0 / DbCL |

Run:

```powershell
python scripts/data/download_benchmark_data.py --all
```

The script downloads only from the official records above, validates that the
payload is not an HTML error page, extracts ZIP archives, and writes checksums.
