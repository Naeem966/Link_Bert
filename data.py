from pykeen.datasets import FB15k237

def load_fb15k237():
    """
    Loads the FB15k-237 dataset using PyKEEN's built-in dataset loader
    and returns the TriplesFactory which holds training, validation, and test splits.
    """
    print("Loading FB15k-237 dataset via PyKEEN...")
    dataset = FB15k237()  # Downloads and preprocesses the dataset
    print("Dataset loaded. Training triples: {}, Validation triples: {}, Testing triples: {}"
          .format(len(dataset.training.triples), len(dataset.validation.triples), len(dataset.testing.triples)))
    return dataset.training, dataset.validation, dataset.testing

if __name__ == "__main__":
    training, validation, testing = load_fb15k237()
    print("Finished loading dataset.")