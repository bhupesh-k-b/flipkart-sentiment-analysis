# Findings

Processed 205,052 public Flipkart reviews and retained 91,814 unique usable texts after cleaning. Removed 964 distinct texts with conflicting sentiment labels. Selected a reproducible stratified sample of 45,000 unique texts.

Compared a majority baseline, Logistic Regression, and a linear SVM using TF-IDF unigrams/bigrams. Selected **logistic_regression** by validation macro F1, refit on the 80% development partition, and evaluated once on the untouched 9,000-review test partition.

- Test accuracy: **86.81%**.
- Test macro F1: **0.7016**.
- Accuracy 95% Wilson interval: 86.10%–87.49%, treating sampled reviews as independent; product clustering may make this optimistic.

The publisher generated labels with VADER and manual checks. This measures agreement with those labels, not independent human judgment. Ratings and product metadata are excluded from model inputs. Exact normalized texts cannot cross partitions; near-duplicates and products can. A product-disjoint or temporal evaluation is a future extension requiring a separate benchmark. Do not substitute an assumed 85% accuracy for the measured result.
