import pandas as pd
import os
import nibabel as nib
import torch

def get_subject_fmri_data(layout, subject: str):
    fmri_files = layout.get(subject=subject, extension='nii.gz', datatype='func', return_type='filename',
                                    task='affordance')

    fmri_imgs = [nib.load(fmri_files[run]) for run in range(len(fmri_files))]
    fmri_data = [fmri_img.get_fdata() for fmri_img in fmri_imgs]
    fmri_data = [x.T.reshape(x.T.shape[0], -1) for x in fmri_data]
    return fmri_data

def get_all_fmri_data(layout, n_samples: int = -1):
    subjects = layout.get_subjects()
    data = []
    if n_samples < 0:
        for subj in subjects:
            data.append(get_subject_fmri_data(layout, subj))
    else:
        for i, subj in enumerate(subjects):
            if i >= n_samples:
                break
            data.append(get_subject_fmri_data(layout, subj))
    return data

def get_stimuli_run(data_path: str, run: int) -> list[tuple[int, int, int, str, str]]:
    """
    Get stimuli used for this run

    Args:
        data_path: path to dataset
        run: run number

    Returns:
        list of tuples consisting of (TR, condition, object 1, object 2)
    """
    # Convert run to gross matlab indexing :p
    run += 1

    # Read the input Excel files (replace with the actual file paths)
    eo_file = os.path.join(data_path, 'stimuli', 'task-affordance_stimuli', 'eo.txt')
    go_file = os.path.join(data_path, 'stimuli', 'task-affordance_stimuli', 'go.txt')
    eo_list = pd.read_csv(eo_file, dtype = str)
    go_list = pd.read_csv(go_file, dtype = str)
    condition_file = os.path.join(data_path, 'code', 'task-affordance_1level', 'conditionOrder_8sequences.xlsx')
    catch_file = os.path.join(data_path, 'code', 'task-affordance_1level', 'catchOrder_8sequences.xlsx')
    order_matrix = pd.read_excel(condition_file, header=None)  # Assuming no headers
    catch_order = pd.read_excel(catch_file, header=None)  # Assuming no headers

    # Configuration Parameters (similar to MATLAB's variables)
    nCat = 16  # Number of catch trials
    nPair = 16  # number of type of object pairs in each nov category
    nNov = 2  # number of nov categories
    nPairLoc = 2  # number of possible location of object pairs
    nInver = 2  # control condition
    nNull = 32  # number of null events
    trsWarm = 4  # number of warm-Ups trs
    trsEnd = 5  # number of ending trs

    # Calculate the total number of trials
    trsTot = trsWarm + trsEnd + (nNov * nPairLoc * nInver) * nPair + nCat + nNull

    # Extract data based on 'run'
    start_tr = (run - 1) * trsTot
    end_tr = run * trsTot

    # Conditions and object pairs
    condition = order_matrix.iloc[start_tr:end_tr, 2]  # Column 3 in MATLAB corresponds to index 2 in Python
    object_pair = order_matrix.iloc[start_tr:end_tr, 3]  # Column 4 in MATLAB corresponds to index 3 in Python
    onset = order_matrix.iloc[start_tr:end_tr, 1]
    catch_condition = catch_order.iloc[1, (run - 1) * nCat:run * nCat]  # Row 2 in MATLAB is row index 1 in Python
    catch_pair = catch_order.iloc[0, (run - 1) * nCat:run * nCat]  # Row 1 in MATLAB is row index 0 in Python

    # Stimuli properties
    tr_to_stim = []
    # Loop through conditions (based on MATLAB's logic)
    for i in range(len(condition)):
        try:
            pair = int(object_pair.iloc[i])
            cond = int(condition.iloc[i])
        except:
            continue
        if pair > 16: # this is to only process familiar objects
            continue
        if cond == 0:  # Condition 0 (do nothing here if needed)
            continue
        elif cond != 9:
            # Parse binary string for condition details
            condition_bin = bin(7 + cond)[2:]  # Convert to binary and pad; equivalent to MATLAB's dec2bin
            condition_bin = condition_bin.zfill(4)  # Pad to 4 bits

            nov = int(condition_bin[2])  # Third character
            inv = int(condition_bin[1])  # Second character
            layout = int(condition_bin[3]) + 1  # Fourth character -> layout (1-based in MATLAB)

            if layout == 1:
                Goname_entry = f"GoM{pair}.jpg"
                if inv == 0:
                    Eoname_entry = f"Eo{pair}.jpg"
                else:
                    Eoname_entry = f"EoR{pair}.jpg"
            else:
                Goname_entry = f"GoMF{pair}.jpg"
                if inv == 0:
                    Eoname_entry = f"EoF{pair}.jpg"
                else:
                    Eoname_entry = f"EoRF{pair}.jpg"

        else:
            # Catch condition processing
            catch_idx = i % len(catch_condition)
            condition_bin = bin(7 + catch_condition.iloc[catch_idx])[2:]
            condition_bin = condition_bin.zfill(4)

            nov = int(condition_bin[2])
            inv = int(condition_bin[1])
            layout = int(condition_bin[3]) + 1

            if layout == 1:
                Goname_entry = f"GoM{pair}.jpg"
                if inv == 0:
                    Eoname_entry = f"Eo{pair}.jpg"
                else:
                    Eoname_entry = f"EoR{pair}.jpg"
            else:
                Goname_entry = f"GoMF{pair}.jpg"
                if inv == 0:
                    Eoname_entry = f"EoF{pair}.jpg"
                else:
                    Eoname_entry = f"EoRF{pair}.jpg"
        tr_to_stim.append((run, onset.iloc[i], cond, eo_list.iloc[pair - 1, 0], go_list.iloc[pair - 1, 0]))
    return tr_to_stim

def get_all_stimuli(data_path: str, runs: int = 8):
    data = []
    for run in range(runs):
        data += get_stimuli_run(data_path, run)
    return pd.DataFrame(data, columns = ['run', 'onset', 'condition', 'object1', 'object2'])

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


