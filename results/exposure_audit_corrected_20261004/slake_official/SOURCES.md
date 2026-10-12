Official SLAKE 1.0 split files (Liu et al. 2021), copied 2026-10-04 from the local release at
<private path redacted> (file dates 2021-03-23).

| file | sha256 | QA rows | distinct img_name |
|---|---|---|---|
| train.json (committed as outputs/legacy_mapping_20260923/source_train_a9083ce6.json, byte-identical) | 7a9fb81f9d7bb145c3d7e9bc8d3b9016ba18b6539ea5031a53bae8fd3cd991c3 | 9835 | 586 |
| validate.json | 32b016440b0c3be11056a78a18eeab46333268407fbb6e6b32f9f4c2debc50f6 | 2099 | 174 |
| test.json | 6be8f7b4c5a46cdbc713a5210a25b6ed5aa1fd1574c83cefb4f998131f17c2c3 | 2094 | 180 |

The distinct-image counts equal the HF BoKelvin/Slake counts recorded in outputs/region_exposure_audit.json
(586/174/180), which is the split source the original generator used.
