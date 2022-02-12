



def flatten_lists_of_lists(list_of_list):
    flat_list = [item for sublist in list_of_list for item in sublist]
    return flat_list