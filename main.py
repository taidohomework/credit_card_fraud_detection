import os

def clear_terminal():
    # Clear the terminal screen
    os.system('cls' if os.name == 'nt' else 'clear')

def preprocess_data():
    clear_terminal()
    print("Starting data preprocessing...")
    os.system("python src/data_preprocessing.py")
    print("Data preprocessing completed.\n")

def train_model():
    clear_terminal()
    print("Starting model training...")
    os.system("python src/model_training.py")
    print("Model training completed.\n")

def evaluate_model():
    clear_terminal()
    print("Starting model evaluation...")
    os.system("python src/model_evaluation.py")
    print("Model evaluation completed.\n")

def generate_random_data():
    clear_terminal()
    n_samples = input("Enter the number of samples to generate: ")
    print(f"Generating {n_samples} random sample data from original dataset...")
    os.system(f"python src/generate_random_data.py {n_samples}")
    print("Random data generation completed.\n")

def make_inference():
    clear_terminal()
    print("Starting inference on new data...")
    os.system("python src/model_inference.py")
    print("Inference completed.\n")

def show_fraud_statistics():
    clear_terminal()
    print("Showing fraud statistics...")
    os.system("python src/fraud_statistics.py")
    input("\nPress Enter to continue...")
    clear_terminal()

def main():
    while True:
        print("Credit Card Fraud Detection - Main Menu")
        print("1. Preprocess Data")
        print("2. Train Model")
        print("3. Evaluate Model")
        print("4. Generate Random Data for Inference")
        print("5. Make Inference on New Data")
        print("6. Show Fraud Statistics")
        print("7. Exit")

        choice = input("Select an option (1-7): ")

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
            show_fraud_statistics()
        elif choice == '7':
            clear_terminal()
            print("Exiting program. Goodbye!")
            break
        else:
            clear_terminal()
            print("Invalid choice. Please select a valid option.\n")

if __name__ == "__main__":
    main()