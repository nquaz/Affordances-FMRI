import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

def get_single_string_embedding(my_string, tokenizer, embedding_layer):
    """Returns the embedding of a single string by averaging the embeddings of its tokens"""
    assert len(my_string) > 0, "Input string is empty"
    assert type(my_string) == str, "Input should be a string"
    tokens = tokenizer(my_string, add_special_tokens=False)['input_ids']
    return embedding_layer(torch.tensor(tokens)).detach().cpu().numpy().mean(0)


def get_multi_string_embedding(my_strings, tokenizer, embedding_layer):
    """Takes in a list of two strings, and returns a single embedding for both jointly by averaging passing them in one order or the other"""
    #FIXME it seems like the way it's being loaded/used now, BERT isn't actually using positional embedding so this isn't strictly necessary
    assert len(my_strings) == 2, "This should only take two strings"
    original_order_str = str.join(' ', my_strings)
    reverse_order_str = str.join(' ', my_strings[::-1])
    original_order_embedding = get_single_string_embedding(original_order_str, tokenizer, embedding_layer)
    reverse_order_embedding = get_single_string_embedding(reverse_order_str, tokenizer, embedding_layer)
    return (original_order_embedding + reverse_order_embedding)/2
