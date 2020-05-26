from ExoRIM.data_generator import SimpleGenerator


def test_simple_generator():
    # test with batch size = 1
    gen = SimpleGenerator()

    # test an epoch and a test phase
    list(gen.training_batch())
    list(gen.test_batch())

    # test with different batch size
    gen = SimpleGenerator(train_batch_size=3, test_batch_size=2, split=0.75)

    # test an epoch and a test phase
    list(gen.training_batch())
    list(gen.test_batch())

if __name__ == "__main__":
    test_simple_generator()