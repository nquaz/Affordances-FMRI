import pandas as pd
import os


def get_stimuli(data_path: str, run: int) -> list[tuple[int, str, str]]:
    """
    Get stimuli used for this run

    Args:
        data_path: path to dataset
        run: run number

    Returns:
        list of stimulus pairs used for this run, each item corresponds to a TR
    """
    # Convert run to gross matlab indexing :p
    run += 1

    # Read the input Excel files (replace with the actual file paths)
    eo_file = os.path.join(data_path, 'stimuli', 'task-affordance_stimuli', 'eo.txt')
    go_file = os.path.join(data_path, 'stimuli', 'task-affordance_stimuli', 'go.txt')
    condition_file = os.path.join(data_path, 'code', 'task-affordance_1level', 'conditionOrder_8sequences.xlsx')
    catch_file = os.path.join(data_path, 'code', 'task-affordance_1level', 'catchOrder_8sequences.xlsx')
    eo_list = pd.read_csv(eo_file)
    go_file = pd.read_csv(go_file)
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
    catch_condition = catch_order.iloc[1, (run - 1) * nCat:run * nCat]  # Row 2 in MATLAB is row index 1 in Python
    catch_pair = catch_order.iloc[0, (run - 1) * nCat:run * nCat]  # Row 1 in MATLAB is row index 0 in Python

    # Stimuli properties
    tr_to_stim = []
    # Loop through conditions (based on MATLAB's logic)
    for i in range(len(condition)):
        pair = object_pair.iloc[i]
        cond = condition.iloc[i]
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
        tr_to_stim.append((cond, Eoname_entry, Goname_entry))
    return tr_to_stim
