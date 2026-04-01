# Custom Land-Cover Lookup Table

By default, terrain-fetcher uses the **GWA4** lookup table bundled with
[windkit](https://windkit.readthedocs.io) to convert ESA WorldCover class IDs
to aerodynamic roughness lengths (z0).

If you need different z0 values — for example to match a site-specific dataset
or a different roughness parameterisation — you can supply your own lookup
table.

---

## Enabling a custom table

In `config.yaml`, set:

```yaml
land_cover_table: custom
```

terrain-fetcher will then look for `landcover_roughness.csv` in the same
directory as the config file.  To use a different path, add:

```yaml
custom_land_cover_table_path: /path/to/my_table.csv
```

---

## File format

The file is plain text (UTF-8).  Each data row must contain at least two
whitespace- or comma-separated columns:

```
class_id   z0   [description ...]
```

| Column | Type | Description |
|--------|------|-------------|
| `class_id` | integer | ESA WorldCover class ID |
| `z0` | float | Aerodynamic roughness length in **metres** |
| `description` | string (optional) | Human-readable label; ignored by the code |

Both comma-separated and whitespace-separated layouts are accepted, so the
following rows are equivalent:

```
10,0.6,Tree cover
10  0.6  Tree cover
```

**Rules:**

- Lines that start with `#` are treated as comments and ignored.
- Blank lines are ignored.
- Both comma-separated and whitespace-separated layouts are accepted.
- If a WorldCover class ID present in the raster is not found in the table,
  the pixel is left at the default fallback value (you can include class `999`
  as a catch-all).

---

## Example

The bundled `landcover_roughness.csv` in the repository root is a ready-to-use
starting point:

```csv
# WorldCover → aerodynamic roughness length (z0) [m]
# class_id,z0,description
10,0.6,Tree cover
20,0.08,Shrubland
30,0.03,Grassland
40,0.07,Cropland
50,0.6,Built-up
60,0.01,Bare / sparse vegetation
70,0.002,Snow and ice
80,0.0002,Permanent water bodies
90,0.08,Herbaceous wetland
95,0.4,Mangroves
100,0.015,Moss and lichen
999,0.05,Unknown / unclassified
```

Copy this file, adjust the z0 values, and point `custom_land_cover_table_path`
at your modified copy.

---

## ESA WorldCover class IDs

For reference, the full ESA WorldCover class list is:

| Class ID | Description |
|----------|-------------|
| 10 | Tree cover |
| 20 | Shrubland |
| 30 | Grassland |
| 40 | Cropland |
| 50 | Built-up |
| 60 | Bare / sparse vegetation |
| 70 | Snow and ice |
| 80 | Permanent water bodies |
| 90 | Herbaceous wetland |
| 95 | Mangroves |
| 100 | Moss and lichen |
