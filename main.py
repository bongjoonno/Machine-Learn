from tests import test_all_models, test_ga_hparam_optimizer

def main() -> list[float, float, float]:
    return test_all_models()


if __name__ == '__main__':
    print(main())