from preprocessing import preprocessor

def main():
    train_data = preprocessor.preprocess_train_data()
    test_data_dict = preprocessor.preprocess_test_data()
        
if __name__ == "__main__":
    main()