from tests import main_model_tests, test_ga_hparam_optimizer

def main() -> list[float, float, float]:
    return main_model_tests()


if __name__ == '__main__':
    print(main())