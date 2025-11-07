from tests.all_tests import run_all_tests

def main():
    return run_all_tests()



if __name__ == '__main__':
    print(main())
    

# to-do
# go through certain number of epochs (every 10) -> 5_000
# optimize lr for each (lr optimizer is complete)
# pick one with lowest loss