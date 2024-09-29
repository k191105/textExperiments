import pandas as pd
import numpy as np
import spacy

nlp = spacy.load('en_core_web_sm') 

pos_tags_list = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
    'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'SCONJ',
    'SYM', 'VERB'
]

pos_to_idx = {pos: idx for idx, pos in enumerate(pos_tags_list)}
num_pos_tags = len(pos_tags_list)

def get_pos_tags(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    return pos_tags

def build_transition_matrix(pos_tags, pos_to_idx, num_pos_tags):
    # Initialize a matrix of zeros
    transition_matrix = np.zeros((num_pos_tags, num_pos_tags))
    # Count transitions
    for i in range(len(pos_tags) - 1):
        current_pos = pos_tags[i]
        next_pos = pos_tags[i + 1]
        if current_pos in pos_to_idx and next_pos in pos_to_idx:
            current_idx = pos_to_idx[current_pos]
            next_idx = pos_to_idx[next_pos]
            transition_matrix[current_idx, next_idx] += 1
    # Normalize rows to get probabilities (transition probabilities)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums
    return transition_matrix

def compute_entropy(transition_matrix):

    epsilon = 1e-10
    transition_matrix = np.where(transition_matrix == 0, epsilon, transition_matrix)

    entropies = -np.sum(transition_matrix * np.log2(transition_matrix), axis=1)
    average_entropy = np.mean(entropies)
    
    return average_entropy

def extract_common_id(doc_id):
    return doc_id.split('@')[0]  



def main(text_type, N='full'):
    df_human = pd.read_parquet('hape-text_human-chunk-2.parquet')
    df_ai = pd.read_parquet('hape-text_gpt-4o-2024-08-06.parquet')

    filtered_df_human = df_human[df_human['doc_id'].str.contains(text_type)].copy()
    filtered_df_ai = df_ai[df_ai['doc_id'].str.contains(text_type)].copy()

    filtered_df_human['common_id'] = filtered_df_human['doc_id'].apply(extract_common_id)
    filtered_df_ai['common_id'] = filtered_df_ai['doc_id'].apply(extract_common_id)

    merged_df = pd.merge(
        filtered_df_human[['common_id', 'text']],
        filtered_df_ai[['common_id', 'text']],
        on='common_id',
        suffixes=('_human', '_ai')
    )

    if N != 'full':
        merged_df = merged_df.head(N)

    print(merged_df.shape)
    entropies_human = []
    entropies_ai = []
    line_numbers = []

    # Process each pair of human and AI texts
    for idx, row in merged_df.iterrows():
        line_number = idx
        # Human text processing
        text_human = row['text_human']
        pos_tags_human = get_pos_tags(text_human)
        transition_matrix_human = build_transition_matrix(pos_tags_human, pos_to_idx, num_pos_tags)
        entropy_human = compute_entropy(transition_matrix_human)
        entropies_human.append(entropy_human)
        # AI text processing
        text_ai = row['text_ai']
        pos_tags_ai = get_pos_tags(text_ai)
        transition_matrix_ai = build_transition_matrix(pos_tags_ai, pos_to_idx, num_pos_tags)
        entropy_ai = compute_entropy(transition_matrix_ai)
        entropies_ai.append(entropy_ai)

        line_numbers.append(line_number)

    data = pd.DataFrame({
        'Line Number': line_numbers,
        'Entropy Human': entropies_human,
        'Entropy AI': entropies_ai
    })

    data.to_csv(f'entropyValues/entropy_{text_type}.csv', index=False)


text_type = 'news'
N = 150 # adjust for more rows
main(text_type, N) # either don't pass N as an argument or pass N = 'full' to use full dataset
