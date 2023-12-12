#!/usr/bin/python

from collections import defaultdict, Counter


def main():
    training_set = "WSJ_02-21.pos"
    development_set = "WSJ_24.pos"
    training_data = merge_files(training_set, development_set, "train_data")
    tagging_data = "WSJ_24.words"

    lines = open_file(training_data)
    test = open_file(tagging_data)

    trans_probs, likelihood_probs = calculate_probs(lines)

    unique_train_word = training_words_to_set(lines)
    unique_development_word = development_words_to_set(test)
    likelihood_probs = handle_oov(unique_train_word, unique_development_word, likelihood_probs)

    likelihood_probs["}"] = {"}": 1}
    likelihood_probs["{"] = {"{": 1}
    likelihood_probs[","] = {",": 1}

    start_prob = trans_probs["B"]

    pos_set = generate_pos_set(trans_probs)

    pos_arr = virtebi(test, pos_set, start_prob, likelihood_probs, trans_probs)

    result = merge_results(test, pos_arr)

    save_to_pos_file(result, "submission.pos")


def open_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines


def save_to_pos_file(data, filename):
    with open(filename, 'w') as file:
        for word, tag in data:
            if word is None:
                file.write('\n')  # Add an empty line for sentence boundary
            else:
                file.write(f"{word}\t{tag}\n")


def calculate_probs(lines):
    transition_table = defaultdict(Counter)
    prev_tag = None
    index_tag_counter = Counter()
    for line in lines:
        line = line.strip()
        if line:  # line not empty
            word, tag = line.split()
            if prev_tag is not None:
                transition_table[prev_tag][tag] += 1
            prev_tag = tag
            index_tag_counter[tag] += 1
        # If empty line : tag =  "B"
        else:
            tag = "B"
            index_tag_counter[tag] += 1
            transition_table[prev_tag][tag] += 1
            prev_tag = "B"

    # Create transition probabilities table
    transition_probabilities = defaultdict(dict)
    for tag, trans_dict in transition_table.items():
        for key, value in trans_dict.items():
            trans_proba = transition_table[tag][key]/index_tag_counter[tag]
            transition_probabilities[tag][key] = trans_proba

    likelihood_table = defaultdict(Counter)
    for line in lines:
        line = line.strip()
        if line:
            word, tag = line.split()
            likelihood_table[tag][word] += 1

    # Create likelihood probabilities table
    likelihood_probs = defaultdict(dict)
    for tag, freq_dict in likelihood_table.items():
        for word, count in freq_dict.items():
            likelihood_prob = count / index_tag_counter[tag]
            likelihood_probs[tag][word] = likelihood_prob

    return transition_probabilities, likelihood_probs


def generate_pos_set(counter):
    set_of_POS = set()
    for pos, count in counter.items():
        set_of_POS.add(pos)
    return set_of_POS


def training_words_to_set(lines):
    unique_training_word_list = set()

    for line in lines:
        line = line.strip()
        if line:
            word, tag = line.split()
            unique_training_word_list.add(word)
    return unique_training_word_list


def development_words_to_set(lines):
    word_set = set()
    for line in lines:
        line = line.strip()
        if line:
            word_set.add(line)

    return word_set


def is_number(s):
    try:
        float(s)
        int(s)
        return True
    except ValueError:
        return False


def handle_oov(words_set, known_set, likelihood_probs):
    pos_set = generate_pos_set(likelihood_probs)
    oov = []
    for word in words_set:
        if word not in known_set:
            oov.append(word)

    for word in oov:
        word_length = len(word)
        if word[0].isupper():
            likelihood_probs["NNP"][word] = 1
        elif word_length >= 2 and word[word_length-2:] == "ss":
            likelihood_probs["NN"][word] = 1
        elif word_length >= 1 and word[word_length-1:] == "s":
            likelihood_probs["NNS"][word] = 1
        elif word_length >= 4 and word[word_length-3:] in ["ish", "ous", "ful", "less", "ble", "ive", "us"]:
            likelihood_probs["JJ"][word] = 1

        pos_set = generate_pos_set(likelihood_probs)
        pos_set = {i for i in pos_set if len(i) > 1}
        pos_set.discard("FW")
        pos_set.discard("TO")

        for key in pos_set:
            likelihood_probs[key][word] = 1/10000
    return likelihood_probs


def virtebi(lines, poset, start_prob, likelihood_probs, trans_probs):
    text = []
    sentence = []
    sentence_pos = []

    tags = list(poset)

    for line in lines:
        word = line.strip()
        if word:
            sentence.append(word)
        else:
            text.append(sentence)
            sentence = []

    for sentence in text:  # for sentence in txt

        # Create 2D array with dimension (len(row), len(columns) )
        virtebi_arr = [[0 for col in range(len(sentence))] for row in range(len(tags))]
        look_up = [[0 for col in range(len(sentence))] for row in range(len(tags))]

        # Initialization step
        for tag in tags:
            ind = tags.index(tag)
            virtebi_arr[ind][0] = float(start_prob.get(tag, 1e-10)) * likelihood_probs[tag].get(sentence[0], 1e-10)

        # Recursion to find max
        for n in range(1, len(sentence)):
            for state in range(len(tags)):
                max_prob = 0
                max_state = 0
                for prev_state in range(1, len(tags)):
                    prob = virtebi_arr[prev_state][n-1] * trans_probs[tags[prev_state]].get(tags[state], 1e-10) * likelihood_probs[tags[state]].get(sentence[n], 1e-10)
                    if prob > max_prob:
                        max_prob = prob
                        max_state = prev_state
                virtebi_arr[state][n] = max_prob
                look_up[state][n] = max_state

        # Backtrack to find best path
        path = []
        current_state = max(range(len(tags)), key=lambda st: virtebi_arr[st][-1])
        for t in range(len(sentence) - 1, -1, -1):
            path.append(tags[current_state])
            current_state = look_up[current_state][t]
        path.reverse()

        sentence_pos.extend(path)

    return sentence_pos


def merge_results(lines, sentence_POS):
    result = []
    pos_index = 0

    for line in lines:
        word = line.strip()
        if word:
            result.append((word, sentence_POS[pos_index]))
            pos_index += 1
        else:
            result.append((None, None))
    return result


def merge_files(f1, f2, f3):
    with open(f1, 'r') as f:
        lines = f.readlines()
    with open(f2, 'r') as f:
        lines2 = f.readlines()
    lines.extend(lines2)
    with open(f3, 'w') as f:
        f.write("".join(lines))
    return f3


if __name__ == "__main__":
    main()
