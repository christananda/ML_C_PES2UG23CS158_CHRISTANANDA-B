import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    label_column = [t[-1].tolist() for t in tensor]
    def calculate_entropy(probs):
        return (-torch.sum(probs * torch.log2(probs)))
    set_label = list(set(label_column))
    counts_probs = []
    for x in set_label:
        counts_probs.append(label_column.count(x) / len(label_column))
    k = (calculate_entropy(torch.tensor(counts_probs))).item()
    return k

def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    current_col_attribute = [t[attribute].tolist() for t in tensor]
    multiple_diff_class = list(set(current_col_attribute))
    label_column = [t[-1].tolist() for t in tensor]
    total_length_of_column = len(current_col_attribute)
    o = 0
    for x in multiple_diff_class:
        feature_count = current_col_attribute.count(x)
        mul_factor_probs = torch.tensor(feature_count / total_length_of_column)
        t1 = []
        t2 = []
        for i in range(len(current_col_attribute)):
            if current_col_attribute[i] == x:
                t1.append(x)
                t2.append(label_column[i])
        new_tensor = torch.cat((torch.tensor(t1).unsqueeze(1), torch.tensor(t2).unsqueeze(1)), dim=1)
        test_entropy = get_entropy_of_dataset(new_tensor)
        if not torch.isnan(torch.tensor(test_entropy)):
            o += (mul_factor_probs * test_entropy).item()
    return o

def get_information_gain(tensor: torch.Tensor, attribute: int):
    return (torch.round(torch.tensor(get_entropy_of_dataset(tensor)) - get_avg_info_of_attribute(tensor, attribute), decimals=4)).item()

def get_selected_attribute(tensor: torch.Tensor):
    gain_info_dictionary = {}
    for i in range(len(tensor[0]) - 1):
        gain_info_dictionary[i] = get_information_gain(tensor, i)
    max_gain = max(gain_info_dictionary.values())
    for i in gain_info_dictionary.keys():
        if gain_info_dictionary[i] == max_gain:
            return (gain_info_dictionary, int(i))
    return ({}, -1)