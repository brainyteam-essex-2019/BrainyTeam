import pickle


def save_model(model, filepath):
    """
    Saves classifiers to pickle file to be read latter
    :param model: classifier model being saved to pickle file
    :param filepath: filepath of file modle is saved to
    :return: true if model was saved successfully, false otherwise
    """
    if model is not None:  # if model supplied is an object
        file = open(filepath, "wb")
        pickle.dump(model, file)
        file.close()
        return True
    return False


def read_model(filepath):
    """
    reads pickle file and returns saved classifier
    :param filepath: filepath of file being read
    :return: classifier read from pickle file
    """
    try:  # checks if file exists
        file = open(filepath, "rb")
        model = pickle.load(file)
        file.close()
        return model
    except IOError:  # if file does no exit
        print("Could not find file", filepath)
        return None
