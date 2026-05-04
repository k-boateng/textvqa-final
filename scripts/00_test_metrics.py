import sys
sys.path.append(".")

from src.metrics import compute_metrics


def main():
    records = [
        {
            "prediction": "nokia",
            "answers": ["nokia", "Nokia phone"],
        },
        {
            "prediction": "coca cola",
            "answers": ["coca-cola", "coke", "coca cola"],
        },
        {
            "prediction": "red",
            "answers": ["blue"],
        },
    ]

    metrics = compute_metrics(records)

    print(metrics)


if __name__ == "__main__":
    main()