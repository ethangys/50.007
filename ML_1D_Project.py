#############
"""PART 1"""
#############

# Read content from files
with open("ES/train", 'rb') as file:
    EStrain_content = file.read()

with open("RU/train", 'rb') as file:
    RUtrain_content = file.read()

with open("ES/dev.in", 'rb') as file:
    ESin_content = file.read()

with open("RU/dev.in", 'rb') as file:
    RUin_content = file.read()

# Convert binary content to UTF-8 encoded lines
estrain = [line.decode('utf-8') for line in EStrain_content.split(b'\n')]
rutrain = [line.decode('utf-8') for line in RUtrain_content.split(b'\n')]
esin = [line.decode('utf-8') for line in ESin_content.split(b'\n')]
ruin = [line.decode('utf-8') for line in RUin_content.split(b'\n')]

def estimate_emissions(data, k=1):
    sentiment_count = {}
    emission_parameters = {}
    
    for line in data:
        if line == "":
            continue
        else:
            word, sentiment = line.rsplit(' ', 1)
            # Count sentiment occurrences
            sentiment_count.setdefault(sentiment, 0)
            sentiment_count[sentiment] += 1
            
            # Build emission_parameters dictionary
            emission_parameters.setdefault(word, {}).setdefault(sentiment, 0)
            emission_parameters[word][sentiment] += 1
    
    # Handle unknown word emissions
    emission_parameters["#UNK#"] = {}
    for sentiment in sentiment_count:
        emission_parameters['#UNK#'][sentiment] = k
        sentiment_count[sentiment] += k
    
    # Divide by count
    for word, sentiment_probs in emission_parameters.items():
        for sentiment, count in sentiment_probs.items():
            sentiment_probs[sentiment] = count / sentiment_count[sentiment]
    
    return emission_parameters

def sentiment_analysis(data):
    params = estimate_emissions(data)
    tag_dict = {}
    for word, sentiment_probs in params.items():
        # Assign the most probable sentiment tag to each word
        most_probable_tag = max(sentiment_probs, key=sentiment_probs.get)
        tag_dict[word] = most_probable_tag
    return tag_dict

# Perform sentiment analysis
estags = sentiment_analysis(estrain)
rutags = sentiment_analysis(rutrain)

def write_to_file(file_path, data_list):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data_list:
            file.write(f"{item}\n")
            
# Write output files for ES
output_es = []
for word in esin:
    if word == "":
        line = ""
    elif word not in estags:
        sentiment = estags["#UNK#"]
        line = f"{word} {sentiment}"
    else:
        sentiment = estags[word]
        line = f"{word} {sentiment}"
    output_es.append(line)

write_to_file("ES/dev.p1.out", output_es)

# Write output files for RU
output_ru = []
for word in ruin:
    if word == "":
        sentiment = ""
    elif word not in rutags:
        sentiment = rutags["#UNK#"]
    else:
        sentiment = rutags[word]
    line = f"{word} {sentiment}"
    output_ru.append(line)

write_to_file("RU/dev.p1.out", output_ru)


#############
"""PART 2"""
#############
import numpy as np

def estimate_transition(data):
    count = {}  # Count occurrences of state transitions
    params = {}  # Store transition probabilities
    states = []  # Store states for each sentence
    sentence = []  # Temporary storage for a sentence
    
    # Parse input data
    for line in data:
        if line != "":
            sentence.append(line.split()[-1])
        elif sentence != [] and line == "":
            states.append(sentence)
            sentence = []
    
    # Calculate counts and probabilities
    for sentence in states:
        if "START" not in count:
            count["START"] = 1
        else:
            count["START"] += 1
    
        if ("START", sentence[0]) not in params:
            params[("START", sentence[0])] = 1
        else:
            params[("START", sentence[0])] += 1
        
        for i in range(len(sentence)):
            if sentence[i] not in count:
                count[sentence[i]] = 1
            else:
                count[sentence[i]] += 1
            if i != 0:
                if (sentence[i-1], sentence[i]) not in params:
                    params[(sentence[i-1], sentence[i])] = 1
                else:
                    params[(sentence[i-1], sentence[i])] += 1
        if "STOP" not in count:
            count["STOP"] = 1
        else:
            count["STOP"] += 1
        if (sentence[-1], "STOP") not in params:
            params[(sentence[-1], "STOP")] = 1
        else:
            params[(sentence[-1], "STOP")] += 1
    
    # Divide by count
    for pair in params:
        params[pair] = params[pair] / count[pair[0]]
    
    return params

# Estimate emissions and transitions for ES and RU training data
es_e_params = estimate_emissions(estrain)
es_t_params = estimate_transition(estrain)
ru_e_params = estimate_emissions(rutrain)
ru_t_params = estimate_transition(rutrain)

def viterbi(e_params, t_params, sentence):
    n = len(sentence)
    path = []  # Store the resulting path
    states = []  # List to hold possible states
    policy = {}  # Dictionary to store optimal state transitions
    
    # Extract states from transition parameters
    for pair in t_params:
        states.append(pair[0])
    states = set(states)  # Convert to set for faster lookup
    
    matrix = [{"START": 0}]  # Initialize the Viterbi matrix with "START" state
    
    # Perform Viterbi algorithm
    for i in range(1, n + 1):
        policy[i] = {}  # Initialize policy for current position
        matrix.append({})
        word = sentence[i - 1]
        
        # Handle unknown words by using "#UNK#" token
        if word not in e_params:
            word = "#UNK#"
        
        for v in e_params[word]:
            matrix[i][v] = -np.inf  # Initialize with negative infinity
            
            for u in matrix[i - 1]:
                if (u, v) not in t_params:
                    continue
                
                # Calculate the score using log probabilities
                score = matrix[i - 1][u] + np.log(e_params[word][v]) + np.log(t_params[(u, v)])
                
                if score > matrix[i][v]:
                    matrix[i][v] = score
                    policy[i][v] = u  # Store the optimal previous state
    
    # Check for the presence of valid paths
    for i in range(n):
        if all(score == -np.inf for score in matrix[i].values()):
            return ["O"] * n  # Return a list of "O" labels if no valid paths
    
    # Find the end state with the maximum score
    end_state = max(matrix[n], key=lambda state: matrix[n][state])
    prev_state = end_state
    path.append(prev_state)  # Append the end state to the path
    
    # Trace back through the policy to find the best path
    for i in range(n, 1, -1):
        path.append(policy[i][prev_state])
        prev_state = policy[i][prev_state]
    
    return path[::-1]  # Return the reversed path as the actual sequence of states

def viterbi_write(input_data, e_params, t_params, output_file):
    sentences = []
    sentence = []
    
    # Split input lines into sentences
    for line in input_data:
        if line != "":
            sentence.append(line)
        elif line == "" and sentence != []:
            sentences.append(sentence)
            sentence = []
    
    # Predict sentiment using Viterbi algorithm for each sentence
    with open(output_file, "w", encoding="utf-8") as file:
        for sentence in sentences:
            path = viterbi(e_params, t_params, sentence)
            
            # Write predicted sentiment for each word
            for word, sentiment in zip(sentence, path):
                line = f"{word} {sentiment}\n"
                file.write(line)
            file.write("\n")

# Process ES dev.in data and write predictions to ES/dev.p2.out
viterbi_write(esin, es_e_params, es_t_params, "ES/dev.p2.out")

# Process RU dev.in data and write predictions to RU/dev.p2.out
viterbi_write(ruin, ru_e_params, ru_t_params, "RU/dev.p2.out")


#############
"""PART 3"""
#############
import math

# Read training data from a file and collect emission and transition counts
def read_file(filepath):
    states = {}
    e_count = {}
    t_count = {}
    train_x = set()

    with open(filepath, 'r', encoding='utf-8') as f:
        yi = "START"
        states["START"] = 1
        for line in f:
            line = line.strip()
            if not line:
                t_count[(yi, "STOP")] = t_count.get((yi, "STOP"), 0) + 1
                states["STOP"] = states.get("STOP", 0) + 1
                yi = "START"
                states["START"] = states.get("START", 0) + 1
                continue

            idx = line.rfind(" ")
            x, yj = line[:idx], line[idx + 1:]
            t_count[(yi, yj)] = t_count.get((yi, yj), 0) + 1
            e_count[(yj, x)] = e_count.get((yj, x), 0) + 1
            states[yj] = states.get(yj, 0) + 1
            train_x.add(x)
            yi = yj

        if yi != "START":
            t_count[(yi, "STOP")] = t_count.get((yi, "STOP"), 0) + 1
            states["STOP"] = states.get("STOP", 0) + 1
        else:
            states["START"] -= 1
    return states, e_count, t_count, train_x

# Calculate emission probabilities based on the given parameters
def e_params(xt, yt, states, e_count, train_x, k=1):
    if xt in train_x:
        if e_count.get((yt, xt), 0) == 0:
            return float("-inf")
        return math.log(e_count.get((yt, xt), 0) / (states[yt] + k))
    else:
        return math.log(k / (states[yt] + k))

# Calculate transition probabilities based on the given parameters
def t_params(yi, yj, t_count, states):
    num = t_count.get((yi, yj), 0)
    denom = states[yi]
    if num == 0:
        return float('-inf')
    return math.log(num / denom)

# Read input sequences from a file
def read_in(file_path):
    x_seq = []
    curr_seq = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            x = line.strip()
            if not x:
                if curr_seq:
                    x_seq.append(curr_seq)
                    curr_seq = []
            else:
                curr_seq.append(x)
        if curr_seq:
            x_seq.append(curr_seq)
    return x_seq

# Write sequence pairs to a file
def write_p_tofile(file_path, seq_list):
    with open(file_path, 'w', encoding='utf-8') as f:
        for seq_pairs in seq_list:
            for x, y in seq_pairs:
                f.write(f"{x} {y}\n")
            f.write("\n")

# Calculate k-best sequences using Viterbi algorithm
def k_best_seq(states, e_count, t_count, train_x, x_input_seq, k=1):
    n = len(x_input_seq)
    s = list(states.keys())

    scores = {}
    for i in range(n + 1):
        for state in s:
            score_list = []
            for j in range(k):
                score_list.append((float('-inf'), None, None))
            scores[(i, state)] = score_list

    scores[(n + 1, "STOP")] = None
    scores[(0, "START")] = [(0.0, None, None)]

    for t in range(1, n + 1):
        for v in s:
            if v == "START" or v == "STOP":
                continue
            all_scores = []
            for u in s:
                if (u == "STOP") or (u == "START" and t != 1):
                    continue
                for idx, scorepair in enumerate(scores[(t - 1, u)]):
                    emission_prob = e_params(x_input_seq[t - 1], v, states, e_count, train_x, 1)
                    transition_prob = t_params(u, v, t_count, states)
                    current_v_score = scorepair[0] + emission_prob + transition_prob
                    all_scores.append((current_v_score, u, idx))
            scores[(t, v)] = sorted(all_scores, reverse=True)[:k]

        all_scores = []
        for u in s:
            if (u == "START") or (u == "STOP"):
                continue
            for idx, scorepair in enumerate(scores[(n, u)]):
                transition_prob = t_params(u, "STOP", t_count, states)
                current_v_score = scorepair[0] + transition_prob
                all_scores.append((current_v_score, u, idx))
        scores[(n + 1, "STOP")] = sorted(all_scores, reverse=True)[:k]

    k_best_paths = []
    score_list = [(n + 1, "STOP")]
    for idx_in_STOP_list in range(k):
        path = []
        score, parent, idx_in_parent = scores[(n + 1, "STOP")][idx_in_STOP_list]
        for i in range(n, 0, -1):
            path.insert(0, parent)
            score, parent, idx_in_parent = scores[(i, parent)][idx_in_parent]
        k_best_paths.append(path)
    return k_best_paths, scores

# Perform computations on input data and write output paths to files
def compute(dev_in_path, states, e_count, t_count, train_x, write_paths, k_paths):
    x_seqs = read_in(dev_in_path)
    k = max(k_paths)
    seqs = {p: [] for p in k_paths}

    for x_seq in x_seqs:
        k_best_paths, scores = k_best_seq(states, e_count, t_count, train_x, x_seq, k)
        for p in k_paths:
            seqs[p].append(list(zip(x_seq, k_best_paths[p - 1])))

    for write_loc, path_num in zip(write_paths, k_paths):
        write_p_tofile(write_loc, seqs[path_num])

# Read training data and compute paths for RU language
states_RU, e_count_RU, t_count_RU, train_x_RU = read_file("RU/train")
compute("RU/dev.in", states_RU, e_count_RU, t_count_RU, train_x_RU, ["RU/dev.p3.2nd.out", "RU/dev.p3.8th.out"], [2, 8])

# Read training data and compute paths for ES language
states_ES, e_count_ES, t_count_ES, train_x_ES = read_file("ES/train")
compute("ES/dev.in", states_ES, e_count_ES, t_count_ES, train_x_ES, ["ES/dev.p3.2nd.out", "ES/dev.p3.8th.out"], [2, 8])


#############
"""PART 4"""
#############
def convert_data_to_lists(data):
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []
    for line in data.split("\n"):
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            token, label = line.rsplit(maxsplit=1)  # Split from the right
            current_sentence.append(token)  # Convert word to lowercase
            current_labels.append(label)
        else:
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)
                current_sentence = []
                current_labels = []

    sentences_lower = [[word for word in sentence] for sentence in sentences]

    return sentences_lower, labels

def viterbi_log(e_params, t_params, sentence):
    n = len(sentence)
    path = []
    states = []  # List to store unique states
    policy = {}  # Dictionary to store the best path policy

    # Extract unique states from transition parameters
    for pair in t_params.keys():
        states.append(pair[0])
    states = set(states)  

    matrix = [{"START": 0}]  # Initialize the Viterbi matrix with the "START" state
    for i in range(1, n + 1):
        policy[i] = {}  # Initialize policy for the current position
        matrix.append({})
        word = sentence[i - 1]
        
        # Handle unknown words by using "#UNK#" token
        if word not in e_params.keys():
            word = "#UNK#"
        
        # Iterate over possible states for the current word
        for v in e_params[word]:
            matrix[i][v] = -np.inf  # Initialize with negative infinity
            for u in matrix[i - 1]:
                if (u, v) not in t_params.keys():
                    continue
                score = matrix[i - 1][u] + e_params[word][v] + t_params[(u, v)]
                if score > matrix[i][v]:
                    matrix[i][v] = score
                    policy[i][v] = u  # Update the best previous state
                    
    # Check for the presence of valid paths
    for i in range(n):
        if all(score == -np.inf for score in matrix[i].values()):
            return ["O"] * n  # Return a list of "O" labels if no valid paths

    # Find the end state with the maximum score
    end_state = max(matrix[n], key=lambda state: matrix[n][state])
    prev_state = end_state
    path.append(prev_state)  # Append the end state to the path
    
    # Trace back through the policy to find the best path
    for i in range(n, 1, -1):
        path.append(policy[i][prev_state])
        prev_state = policy[i][prev_state]
    
    return path[::-1]  # Return the reversed path as the actual sequence of states

with open("ES/train", 'rb') as file:
    EStrain_content = file.read().decode("utf-8")

with open("RU/train", 'rb') as file:
    RUtrain_content = file.read().decode("utf-8")

logged_es_t_values = {}
for key, value in es_t_params.items():
    logged_es_t_values[key] = 0

logged_es_e_values = {}
for key, inner_dict in es_e_params.items():
    logged_es_e_values[key] = {}
    for inner_key, inner_value in inner_dict.items():
        logged_es_e_values[key][inner_key] = 0

logged_ru_t_values = {}
for key, value in ru_t_params.items():
    logged_ru_t_values[key] = 0     

logged_ru_e_values = {}
for key, inner_dict in ru_e_params.items():
    logged_ru_e_values[key] = {}
    for inner_key, inner_value in inner_dict.items():
        logged_ru_e_values[key][inner_key] = 0

def train_struct_perceptron_with_smoothing(e_params, t_params, data, r, epochs, smoothing_factor=1):
    sentences, labels = convert_data_to_lists(data)
    
    for k in range(epochs):
        print(k)
        for i in range(len(sentences)):
            predicted_path = viterbi_log(e_params, t_params, sentences[i])
            actual_path = labels[i]

            updated_e_params = {word: {label: 0 for label in label_set} for word, label_set in e_params.items()}
            updated_t_params = {label_pair: 0 for label_pair in t_params.keys()}
            
            for j in range(len(predicted_path)):
                if predicted_path[j] != actual_path[j]:
                    if j != 0 and j != (len(predicted_path) - 1):
                        updated_t_params[(actual_path[j-1], actual_path[j])] += r
                        updated_t_params[(actual_path[j], actual_path[j+1])] += r
                        updated_e_params[sentences[i][j]][actual_path[j]] += r
                        
                        updated_t_params[(predicted_path[j-1], predicted_path[j])] -= r
                        updated_t_params[(predicted_path[j], predicted_path[j+1])] -= r
                        updated_e_params[sentences[i][j]][predicted_path[j]] -= r

                    if j == 0:
                        if ("START", actual_path[j]) not in t_params:
                            updated_t_params[("START", actual_path[j])] = 0
                        updated_t_params[("START", actual_path[j])] += r
                        updated_t_params[(actual_path[j], actual_path[j+1])] += r
                        updated_e_params[sentences[i][j]][actual_path[j]] += r
        
                        if ("START", predicted_path[j]) not in t_params:
                            updated_t_params[("START", predicted_path[j])] = 0
                        updated_t_params[("START", predicted_path[j])] -= r
                        updated_t_params[(predicted_path[j], predicted_path[j+1])] -= r
                        updated_e_params[sentences[i][j]][predicted_path[j]] -= r
                    
                    if j == (len(predicted_path)-1):
                        updated_t_params[(actual_path[j-1], actual_path[j])] += r
                        if (actual_path[j], "STOP") not in t_params:
                            updated_t_params[(actual_path[j], "STOP")] = 0
                        updated_t_params[(actual_path[j], "STOP")] += r
                        updated_e_params[sentences[i][j]][actual_path[j]] += r
                        updated_t_params[(predicted_path[j-1], predicted_path[j])] -= r
                        if (predicted_path[j], "STOP") not in t_params:
                            updated_t_params[(predicted_path[j], "STOP")] = 0
                        updated_t_params[(predicted_path[j], "STOP")] -= r
                        updated_e_params[sentences[i][j]][predicted_path[j]] -= r
                            
            
            for word in updated_e_params:
                for label in updated_e_params[word]:
                    e_params[word][label] += updated_e_params[word][label]
                    
            for label_pair in updated_t_params:
                if label_pair not in t_params:
                    t_params[label_pair]=0
                t_params[label_pair] += updated_t_params[label_pair]
    
    return e_params, t_params

ru_new_weights_e,ru_new_weights_t=train_struct_perceptron_with_smoothing(logged_ru_e_values,logged_ru_t_values, RUtrain_content,0.1,15)
es_new_weights_e, es_new_weights_t=train_struct_perceptron_with_smoothing(logged_es_e_values,logged_es_t_values, EStrain_content,0.1,20)

def struc_perceptron_write(input_lines, new_weights_e, new_weights_t, output_file):
    sentences = []
    sentence = []

    for line in input_lines:
        if line.strip():
            sentence.append(line.strip())
        elif sentence:
            sentences.append(sentence)
            sentence = []

    with open(output_file, "w", encoding='utf-8') as file:
        for sentence in sentences:
            path = viterbi_log(new_weights_e, new_weights_t, sentence)
            for word, sentiment in zip(sentence, path):
                line = f"{word} {sentiment}\n"
                file.write(line)
            file.write("\n")

struc_perceptron_write(esin, es_new_weights_e, es_new_weights_t, "ES/dev.p4.out")

struc_perceptron_write(ruin, ru_new_weights_e, ru_new_weights_t, "RU/dev.p4.out")

with open("ES/test.in", "rb",) as file:
    ES_test_content = file.read()
estest = [line.decode('utf-8') for line in ES_test_content.split(b'\n')]

struc_perceptron_write(estest, es_new_weights_e, es_new_weights_t, "ES/test.p4.out")

with open("RU/test.in", "rb",) as file:
    RU_test_content = file.read()
rutest = [line.decode('utf-8') for line in RU_test_content.split(b'\n')]

struc_perceptron_write(rutest, ru_new_weights_e, ru_new_weights_t, "RU/test.p4.out")