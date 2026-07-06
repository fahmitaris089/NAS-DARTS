# Prediction Overlap Summary

## Accuracy
- c12: 99.52%
- mobilenet: 99.64%
- effv2m: 100.00%
- resnet50: 100.00%

## C12 Error Analysis
- C12 error count: 4
- C12 errors where EfficientNetV2M and ResNet50 are correct: 4
- C12 errors where MobileNetV3Small is correct: 3
- C12 errors where true label is in C12 top-5: 3
- C12 errors with low margin: 3
- C12 errors wrong for all compared models: 0

## Recommendations
- Teachers solve all C12 errors; prioritize hard-sample KD, margin KD, or top-k distillation.
- MobileNet solves some C12 errors; compare MobileNet/C12 features or try multi-teacher KD.
- Some true labels are in C12 top-5; margin-ranking or ArcFace/SupCon fine-tuning is promising.
