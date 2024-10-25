import os
import pandas as pd

def preprocess_data():
    print("Starting data preprocessing...")
    os.system("python src/data_preprocessing.py")
    print("Data preprocessing completed.\n")

def train_model():
    print("Starting model training...")
    os.system("python src/model_training.py")
    print("Model training completed.\n")

def evaluate_model():
    print("Starting model evaluation...")
    os.system("python src/model_evaluation.py")
    print("Model evaluation completed.\n")

def generate_random_data():
    print("Generating random sample data from original dataset...")
    os.system("python src/generate_random_data.py")

def make_inference():
    print("Starting inference on new data...")
    os.system("python src/model_inference.py")
    print("Inference completed.\n")

def main():
    while True:
        print("Credit Card Fraud Detection - Main Menu")
        print("1. Preprocess Data")
        print("2. Train Model")
        print("3. Evaluate Model")
        print("4. Generate Random Data for Inference")
        print("5. Make Inference on New Data")
        print("6. Exit")

        choice = input("Select an option (1-6): ")

        if choice == '1':
            preprocess_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            evaluate_model()
        elif choice == '4':
            generate_random_data()
        elif choice == '5':
            make_inference()
        elif choice == '6':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please select a valid option.\n")

if __name__ == "__main__":
    main()
