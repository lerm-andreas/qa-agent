"""Train the intent classifier once and persist it

Run: `python train_intent.py`
Produces ml/intent_classifier.joblib (gitignored). The agent and the comparison
demo load this artifact; they do not retrain on startup.
"""
from collections import Counter

from ml.intent_classifier import DEFAULT_MODEL_PATH, IntentClassifier
from ml.intent_data import TEST_DATA, TRAINING_DATA


def main() -> None:
    print(f"Training on {len(TRAINING_DATA)} examples "
          f"({dict(Counter(label for _, label in TRAINING_DATA))})")

    classifier = IntentClassifier(model_path=None)  # start unfitted
    classifier.train(TRAINING_DATA)
    classifier.save(DEFAULT_MODEL_PATH)
    print(f"Saved model → {DEFAULT_MODEL_PATH}")

    correct = sum(classifier.predict(q)[0] == gold for q, gold in TEST_DATA)
    print(f"Held-out accuracy: {correct}/{len(TEST_DATA)} = {correct / len(TEST_DATA) * 100:.1f}%")


if __name__ == "__main__":
    main()
