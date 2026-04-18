"""Train regression on SQL orders and save joblib model + print metrics."""
import database as db
import ml_model


def main():
    db.init_db()
    result = ml_model.train_and_save()
    print(f"Trained on {result.train_rows} rows.")
    print(f"Hold-out MAE:  {result.mae:.3f} minutes")
    print(f"Hold-out RMSE: {result.rmse:.3f} minutes")


if __name__ == "__main__":
    main()
