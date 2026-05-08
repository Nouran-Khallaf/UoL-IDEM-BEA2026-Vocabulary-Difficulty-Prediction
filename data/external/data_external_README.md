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

Used for cognate and cross-lingual form-overlap features.

Source repository:

```text
https://github.com/kbatsuren/CogNet
```

Download CogNet v2.0:

```text
https://github.com/kbatsuren/CogNet/raw/master/CogNet-v2.0.zip
```

After downloading, unzip the archive and place/rename the TSV file as:

```text
data/external/CogNet-v2.0.tsv
```

CogNet is distributed by its authors under CC BY-NC-SA 4.0. Please check the original repository before redistribution or reuse.

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
