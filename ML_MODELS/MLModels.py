# run_all_models.py
import os
import importlib.util


def run_model(file_name):
    """Execute a Python file directly"""
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {file_name}")
    print(f"{'=' * 60}")

    try:
        # Read the file content
        with open(file_name, 'r', encoding='utf-8') as file:
            code = file.read()

        # Execute the code
        exec(code)
        return True
    except Exception as e:
        print(f"Error running {file_name}: {str(e)}")
        return False


def main():
    # List of all model files to run
    model_files = [
        "Liner_regression.py",
        "Lasso_Regression.py",
        "Decision_Tree.py",
        "Random_forest.py",
        "KNN_Regression.py",
        "Gradient_Boosting.py",
        "Logistic_Regression.py",
        "Gaussian Naive Bayes.py",
        "SVM.py",
        "XGBoost.py"
    ]

    print("Starting AQI Model Evaluation")
    print("This will run all AQI prediction models and display their results")

    # Run each model file
    for file_name in model_files:
        if os.path.exists(file_name):
            run_model(file_name)
        else:
            print(f"\n❌ File not found: {file_name}")

    print("\nAll models have been executed.")


if __name__ == "__main__":
    main()