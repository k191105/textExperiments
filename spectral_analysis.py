import pandas as pd
import numpy as np
import spacy

nlp = spacy.load('en_core_web_sm') 



# taken from spacy documentation
pos_tags_list = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
    'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'SCONJ',
    'SYM', 'VERB', 'PUNCT', 'X', 'SPACE'
]

# label each pos
pos_to_idx = {pos: idx for idx, pos in enumerate(pos_tags_list)}

print(pos_to_idx)
num_pos_tags = len(pos_tags_list)


def get_pos_tags(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    return pos_tags

def build_transition_matrix(pos_tags, pos_to_idx, num_pos_tags):
    # build numPosTags by numPosTags matrix
    transition_matrix = np.zeros((num_pos_tags, num_pos_tags))

    for i in range(len(pos_tags) - 1):
        current_pos = pos_tags[i]
        next_pos = pos_tags[i + 1]
        if (current_pos in pos_to_idx) and (next_pos in pos_to_idx):
            current_idx = pos_to_idx[current_pos]
            next_idx = pos_to_idx[next_pos]
            transition_matrix[current_idx, next_idx] += 1

    row_sums = transition_matrix.sum(axis=1, keepdims=True)

    # print(row_sums)
    # making sure that we never divide by 0
    row_sums[row_sums == 0] = 1

    transition_matrix = transition_matrix / row_sums

    # print("transition matrix \n", transition_matrix, "\n")
    return transition_matrix

def compute_spectral_gap(transition_matrix):
    eigenvalues = np.linalg.eigvals(transition_matrix)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    spectral_gap = eigenvalues[0] - eigenvalues[1]
    return spectral_gap

def extract_common_id(doc_id):
    return doc_id.split('@')[0]  


def main(text_type, N='full'):
    df_human = pd.read_parquet('hape-text_human-chunk-2.parquet')
    df_ai = pd.read_parquet('hape-text_gpt-4o-2024-08-06.parquet')


    filtered_df_human = df_human[df_human['doc_id'].str.contains(text_type)].copy()
    filtered_df_ai = df_ai[df_ai['doc_id'].str.contains(text_type)].copy()

    filtered_df_human['common_id'] = filtered_df_human['doc_id'].apply(extract_common_id)
    filtered_df_ai['common_id'] = filtered_df_ai['doc_id'].apply(extract_common_id)

    # merge the datasets for direct comparison
    merged_df = pd.merge(
        filtered_df_human[['common_id', 'text']],
        filtered_df_ai[['common_id', 'text']],
        on='common_id',
        suffixes=('_human', '_ai')
    )
    
    if N != 'full':
        merged_df = merged_df.head(N)
    
    print(merged_df.shape)

    spectral_gaps_human = []
    spectral_gaps_ai = []
    line_numbers = []

    for idx, row in merged_df.iterrows():
        line_number = idx
        # Human stuff
        text_human = row['text_human']
        pos_tags_human = get_pos_tags(text_human)
        transition_matrix_human = build_transition_matrix(pos_tags_human, pos_to_idx, num_pos_tags)
        spectral_gap_human = compute_spectral_gap(transition_matrix_human)
        spectral_gaps_human.append(spectral_gap_human)

        # AI stuff
        text_ai = row['text_ai']
        pos_tags_ai = get_pos_tags(text_ai)
        transition_matrix_ai = build_transition_matrix(pos_tags_ai, pos_to_idx, num_pos_tags)
        spectral_gap_ai = compute_spectral_gap(transition_matrix_ai)
        spectral_gaps_ai.append(spectral_gap_ai)


        line_numbers.append(line_number)

    data = pd.DataFrame({
        'Line Number': line_numbers,
        'Spectral Gap Human': spectral_gaps_human,
        'Spectral Gap AI': spectral_gaps_ai
    })

    data.to_csv(f'spectralGaps/spectral_gaps_{text_type}.csv', index=False)

text_type = 'blog'
N = 150 # adjust for more rows
main(text_type, N) # either don't pass N as an argument or pass N = 'full' to use full dataset
