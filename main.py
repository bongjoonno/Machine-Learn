from tests import main_model_tests, ga_hyperparameter_optimizer_test

def main() -> list[float, float, float]:
    return main_model_tests()


if __name__ == '__main__':
    print(main())