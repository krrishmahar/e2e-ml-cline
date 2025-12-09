# ML Best Practices

## 1. Data
- Validate schema and data types.
- Store provenance metadata.

## 2. Models
- Version all models.
- Log metrics (accuracy, precision, recall, ROC AUC).

## 3. Training
- Deterministic runs using seeds.
- Log training curves.

## 4. Deployment
- Add drift detection in Oumi.
- Raise Kestra event on drop in accuracy > 5%.

