from preprocessing import preprocessor

# Toggle verbose for added commentary
def main():
    train_data_dict = preprocessor.preprocess_train_data()
    test_data_dict = preprocessor.preprocess_test_data()
        
if __name__ == "__main__":
    main()