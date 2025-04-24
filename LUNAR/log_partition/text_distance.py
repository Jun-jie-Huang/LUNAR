import numpy as np
import re
import multiprocessing
from collections import defaultdict, Counter


def longest_common_sequence_words(str1, str2):
    words1, words2 = str1.split(), str2.split()

    dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]
    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_num = dp[len(words1)][len(words2)]

    lcs = []
    i, j = len(words1), len(words2)
    while i > 0 and j > 0:
        if words1[i - 1] == words2[j - 1]:
            lcs.append(words1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    lcs.reverse()
    lcs_str = ' '.join(lcs)

    return lcs_str, lcs_num


def longest_common_string_words(str1, str2):
    words1, words2 = str1.split(), str2.split()

    dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]
    # length and end position of lcs
    max_length = 0
    end_position = 0
    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_position = i

    start_position = end_position - max_length
    longest_common_substring = ' '.join(words1[start_position:end_position])

    return longest_common_substring, max_length


def distance_longest_common_sequence_words(str1, str2):
    words1, words2 = str1.split(), str2.split()
    dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]

    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_num = dp[len(words1)][len(words2)]

    return lcs_num


def distance_longest_common_string_words(str1, str2):
    words1, words2 = str1.split(), str2.split()
    dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]

    max_length = 0
    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]

    return max_length


def distance_hamming_words(str1, str2):
    """ Before applying, make sure the length of str1 and str2 are the same."""
    words1, words2 = str1.split(), str2.split()
    return sum([words1[i] != words2[i] for i in range(len(words1))])


def distance_edit_words(str1, str2):
    words1, words2 = str1.split(), str2.split()
    dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]
    for i in range(len(words1) + 1):
        dp[i][0] = i
    for j in range(len(words2) + 1):
        dp[0][j] = j

    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j]) + 1

    return dp[len(words1)][len(words2)]


def similarity_jaccard_words(str1, str2):
    words1, words2 = set(str1.split()), set(str2.split())

    return len(words1 & words2) / len(words1 | words2)


def calculate_jaccard_similarity(text_list):
    n = len(text_list)
    similarity_matrix = [[0] * n for _ in range(n)]

    def calculate_similarity(i, j):
        similarity = similarity_jaccard_words(text_list[i], text_list[j])
        similarity_matrix[i][j] = similarity
        similarity_matrix[j][i] = similarity

    pool = multiprocessing.Pool()
    for i in range(n):
        for j in range(i + 1, n):
            pool.apply_async(calculate_similarity, args=(i, j))
    pool.close()
    pool.join()
    similarity_matrix = np.array(similarity_matrix)

    return similarity_matrix



def calculate_jaccard_one_to_many(str_anchor, text_list):
    words_anchor = set(str_anchor.split())
    words_list = [set(text.split()) for text in text_list]
    similarity_list = [len(words_anchor & i) / len(words_anchor | i) for i in words_list]

    return similarity_list


def calculate_jaccard_one_to_many_mask(str_anchor, text_list):
    words_anchor = set(str_anchor.split())
    words_list = [set(text.split()) for text in text_list]
    words_counter = Counter(list(words_anchor) + [word for words in words_list for word in words])
    masks = {i: judge_bad_word(i) for i in words_counter.keys()}
    words_anchor = set(i for i in words_anchor if not masks[i])
    words_list = [set(i for i in words if not masks[i]) for words in words_list]
    words_list = [words for words in words_list if len(words) > 0]
    similarity_list = [len(words_anchor & i) / len(words_anchor | i) for i in words_list]

    return similarity_list


def calculate_jaccard_self_loop(text_list):
    all_similarities = [calculate_jaccard_one_to_many(text_list[i], text_list[i+1:]) for i in range(len(text_list)-1)]
    all_similarities = [item for sublist in all_similarities for item in sublist]

    # return sum(all_similarities) / len(all_similarities)
    return all_similarities


def calculate_jaccard_and_diff_self_all_comp(text_list, lamb=0.5):
    if len(text_list) == 0 or len(text_list) == 1:
        return 0
    all_similarities = [calculate_jaccard_one_to_many(text_list[i], text_list[i+1:]) for i in range(len(text_list)-1)]
    all_similarities = [item for sublist in all_similarities for item in sublist]
    # mean_div = sum([(x - y) ** 2 for x in all_similarities for y in all_similarities]) / (len(all_similarities)**2)
    mean_div = sum([abs(x - y) for x in all_similarities for y in all_similarities]) / (len(all_similarities)**2)
    mean_diffs = sum(all_similarities) / len(all_similarities)

    # return lamb * mean_diffs + (1 - lamb) * mean_div, mean_diffs, mean_div
    return lamb * mean_diffs + (1 - lamb) * mean_div


def calculate_jaccard_and_diff_self_loop_all_first(text_list, lamb=0.5):
    if len(text_list) == 0 or len(text_list) == 1:
        return 0
    all_similarities = calculate_jaccard_one_to_many(text_list[0], text_list[1:])
    # mean_div = sum([(x - y) ** 2 for x in all_similarities for y in all_similarities]) / (len(all_similarities)**2)
    mean_div = sum([abs(x - y) for x in all_similarities for y in all_similarities]) / (len(all_similarities)**2)
    mean_diffs = sum(all_similarities) / len(all_similarities)

    # return lamb * mean_diffs + (1 - lamb) * mean_div, mean_diffs, mean_div
    return lamb * mean_diffs + (1 - lamb) * mean_div


def calculate_jaccard_and_diff_self_loop_first_comp(text_list, lamb=0.5):
    if len(text_list) == 0 or len(text_list) == 1:
        return 0
    all_similarities = calculate_jaccard_one_to_many(text_list[0], text_list[1:])
    mean_diffs = sum(all_similarities) / len(all_similarities)

    all_similarities = [calculate_jaccard_one_to_many(text_list[i], text_list[i+1:]) for i in range(len(text_list)-1)]
    all_similarities = [item for sublist in all_similarities for item in sublist]
    mean_div = sum([abs(x - y) for x in all_similarities for y in all_similarities]) / (len(all_similarities)**2)

    # return lamb * mean_diffs + (1 - lamb) * mean_div, mean_diffs, mean_div
    return lamb * mean_diffs + (1 - lamb) * mean_div


def judge_bad_word(word):
    has_digit = bool(re.search(r'\d', word))
    has_symbol = bool(re.search(r'[^\w]', word))
    has_letter = bool(re.search(r'[a-zA-Z]', word))

    return has_digit and has_symbol and has_letter


def calculate_same_one_to_many(str_anchor, text_list):
    words_anchor = set(str_anchor.split())
    words_list = [set(text.split()) for text in text_list]
    similarity_list = [len(words_anchor & i) for i in words_list]

    return similarity_list