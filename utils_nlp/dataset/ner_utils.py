def get_sentence_and_labels(text, data_type="", join_characeter=" "):
    """
    Helper function converting data in conll format to sentence and list
    of token labels.

    Args:
        text (str): Text string in conll format, e.g.
            "Amy B-PER
             ADAMS I-PER
             works O
             at O
             the O
             University B-ORG
             of I-ORG
             Minnesota I-ORG
             . O"
        data_type (str, optional): String that briefly describes the data,
            e.g. "train"
    Returns:
        tuple:
            (list of sentences, list of token label lists)
    """
    text_list = text.split("\n\n")
    if text_list[-1] in (" ", ""):
        text_list = text_list[:-1]

    max_seq_len = 0
    sentence_list = []
    labels_list = []
    for s in text_list:
        # split each sentence string into "word label" pairs
        s_split = s.split("\n")
        # split "word label" pairs
        s_split_split = [t.split() for t in s_split]
        sentence_list.append(
            " ".join([t[0] for t in s_split_split if len(t) > 1])
        )
        labels_list.append([t[1] for t in s_split_split if len(t) > 1])
        if len(s_split_split) > max_seq_len:
            max_seq_len = len(s_split_split)
    print(
        "Maximum sequence length in {0} data is: {1}".format(
            data_type, max_seq_len
        )
    )
    return sentence_list, labels_list