# External resources

This directory is used for external lexical and frequency resources required by some feature-building scripts. These files are **not redistributed** with this repository because they may have separate licences or distribution restrictions.

Download the resources from their original providers and place them in this directory using the filenames expected by the scripts.

## Expected files

```text
data/external/
├── CogNet-v2.0.tsv
├── SUBTLEX-US.xlsx
└── en_m3.xls
```

## Resources

### CogNet

Used for cognate/link-based features.

Place the file here:

```text
data/external/CogNet-v2.0.tsv
```

### SUBTLEX-US

Used for subtitle-frequency features such as word frequency and contextual diversity.

Place the spreadsheet here:

```text
data/external/SUBTLEX-US.xlsx
```

### English frequency / word-list resource

Used for additional frequency-based features in the original experiments.

Place the spreadsheet here:

```text
data/external/en_m3.xls
```

## Notes

- Keep the filenames exactly as shown above, or update the corresponding paths in the configuration file.
- Do not commit downloaded external resources to GitHub.
- If a resource is unavailable, disable the corresponding feature group in the config or remove those features from the selected feature list.
- The repository should contain this README only inside `data/external/`, not the external data files themselves.
