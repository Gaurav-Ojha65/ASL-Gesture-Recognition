# Internal Test Set Classification Report
Model: models/asl_landmark_mlp_best.pt
## Evaluation
```text
              precision    recall  f1-score   support

           A       0.83      0.48      0.61       145
           B       0.98      0.92      0.95        60
           C       0.83      1.00      0.91        79
           D       0.56      0.93      0.70       164
           E       0.96      0.93      0.95       107
           F       0.98      1.00      0.99       450
           G       0.98      0.94      0.96       249
           H       0.80      0.85      0.82        78
           I       0.96      0.76      0.85        84
           J       0.99      0.99      0.99       222
           K       1.00      1.00      1.00       293
           L       0.99      0.96      0.97       244
           M       0.48      0.96      0.64        93
           N       0.85      0.90      0.88        78
           O       0.75      0.37      0.49       182
           P       1.00      0.46      0.63        28
           Q       0.98      0.99      0.99       108
           R       0.93      0.35      0.51       226
           S       1.00      0.91      0.95       265
           T       0.87      0.63      0.73        71
           U       0.57      0.98      0.72       178
           V       0.99      0.97      0.98       240
           W       0.99      1.00      1.00       239
           X       0.93      0.87      0.90        46
           Y       0.99      0.92      0.95       162
           Z       0.96      0.96      0.96       119
      DELETE       0.81      1.00      0.89       102
       SPACE       0.71      0.94      0.81        87

    accuracy                           0.88      4399
   macro avg       0.88      0.86      0.85      4399
weighted avg       0.90      0.88      0.87      4399

```

> **Important Note:** Per-class metrics based on very small support have greater statistical uncertainty. Do not convert this into claims about why those classes behave differently.