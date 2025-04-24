import regex as re
import sys
sys.path.append("../")
from LUNAR.llm_module.post_process import post_process_template
from LUNAR.utils import validate_template, verify_template_for_log_with_first_token


alphabet_set = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
smallletter_set = set("abcdefghijklmnopqrstuvwxyz")
bigletter_set = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def split_template(template, split=[" "]):
    """
    Split a log template into parts based on the specified split characters.

    :param template: The log template to be split.
    :param split: A list of characters to split the template by. Defaults to [" "].
    :return: The split parts of the template.
    """
    pattern = '|'.join(re.escape(s) for s in split)
    result = re.split(pattern, template)
    result = [part for part in result if part]
    return result


def split_template_naive(template):
    """
    Split a log template into parts using the default space character.

    :param template: The log template to be split.
    :return: A list of parts obtained by splitting the template.
    """
    return template.split(" ")


def jaccard_similarity(parts1, parts2):
    """
    Calculate the Jaccard similarity between two sets of template parts.

    :param parts1: The first set of template parts.
    :param parts2: The second set of template parts.
    :return: The Jaccard similarity score.
    """
    common = set(parts1).intersection(parts2)
    union = set(parts1).union(parts2)
    return (len(common) + 0.00001) / (len(union) + 0.00001)


def weighted_jaccard_similarity(parts1, parts2, N=20):
    """
    Calculate the weighted Jaccard similarity between two sets of template parts.

    :param parts1: The first set of template parts.
    :param parts2: The second set of template parts.
    :param N: The weight parameter. Defaults to 20.
    :return: The weighted Jaccard similarity score.
    """
    len1, len2 = len(parts1), len(parts2)
    # N = max(len1, len2)
    # N = max(len1, len2)
    weight = (len1 * len2 / (N ** 2)) ** 2
    return jaccard_similarity(parts1, parts2) * weight


def edit_distance(parts1, parts2):
    m, n = len(parts1), len(parts2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                if parts1[i - 1] == parts2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def common_prefix(str1, str2):
    min_length = min(len(str1), len(str2))
    common_chars = []
    for i in range(min_length):
        if str1[i] == str2[i]:
            common_chars.append(str1[i])
        else:
            break

    return ''.join(common_chars)


def common_suffix(str1, str2):
    """
    Find the common suffix between two strings.

    :param str1: The first string.
    :param str2: The second string.
    :return: The common suffix of the two strings.
    """
    min_length = min(len(str1), len(str2))
    common_chars = []
    for i in range(1, min_length + 1):
        if str1[-i] == str2[-i]:
            common_chars.append(str1[-i])
        else:
            break

    common_chars.reverse()
    return ''.join(common_chars)


def merge_template_by_star(template1, template2, split=[" "]):
    """
    Merge two templates using the '*' placeholder.

    :param template1: The new template.
    :param template2: The old template in the template database.
    :param split: A list of characters to split the templates by. Defaults to [" "].
    :return: The merged template.
    """
    # parts1: new template, parts2: old template in template database
    parts1 = split_template(template1, split)
    parts2 = split_template(template2, split)

    if parts1 == parts2:
        return template1, True
    if len(parts1) == len(parts2):
        common, edit = 0, 0
        new_parts = []
        for part1, part2 in zip(parts1, parts2):
            if part1 == part2:
                common += 1
                new_parts.append(part1)
            else:
                # new_part = greedy_merge_two_vars(part1, part2)
                new_part = greedy_merge_two_vars_both_side(part1, part2)
                if new_part:
                    new_parts.append(new_part)
                    edit += 1
                else:
                    new_parts.append(part1)
        if edit == 0:
            return template1, False
        new_template = ' '.join(new_parts)
        new_template = post_process_template(new_template, [])[0]
        if not validate_template(new_template):
            return template1, False
        if not verify_template_for_log_with_first_token(template1, new_template) or not verify_template_for_log_with_first_token(template2, new_template):
            return template1, False
        return new_template, True
    return template1, False


def judge_var_token_naive(token1, token2, placeholder="\0"):
    alphabet_set = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if set(token1).issubset(alphabet_set) or set(token2).issubset(alphabet_set):
        return False
    else:
        return True


def parenthesis_match(str1, str2, placeholder="\0"):
    if str1 == "(<*>)" and "(" in str2 and  ")" in str2:
        return True
    return False


def colon_token(str1, placeholder="\0"):
    if str1[-1] == ":":
        return True
    return False


def func_token(str1, placeholder="\0"):
    if str1[-2:] == "()":
        return True
    return False


def is_a_var_token(token, tokenset):
    dif_token = tokenset.difference(alphabet_set)
    dif_token_big = tokenset.difference(bigletter_set)
    if len(dif_token_big) == 0:
        return True
    # elif len(dif_token_big) == 1 and "_" in dif_token_big:
    #     return True
    if len(dif_token) == 0:
        return False
    elif len(dif_token) == 1:
        if "." in dif_token or "_" in dif_token:
            return False
        elif colon_token(token):
            return False
    elif len(dif_token) == 2:
        if func_token(token):
            return False
    if "....." in token:
        return False
    return True


def judge_var_token_all(token1, token2, placeholder="\0"):
    """
    return True is both token1 and token2 are variables, else False if one of them is not like a variable should be merged

    """
    token1 = token1.replace("<*>", placeholder)
    token2 = token2.replace("<*>", placeholder)
    token1set, token2set = set(token1), set(token2)

    dif_token1, dif_token2 = token1set.difference(alphabet_set), token2set.difference(alphabet_set)
    if is_a_var_token(token1, token1set) and is_a_var_token(token2, token2set):
        return True
    if parenthesis_match(token1, token2set, placeholder) or parenthesis_match(token2, token1set, placeholder):
        return True

    return False


def judge_var_token(token1, token2, placeholder="\0"):
    """
    return True is both token1 and token2 are variables, else False if one of them is not like a variable should be merged

    """
    token1 = token1.replace("<*>", placeholder)
    token2 = token2.replace("<*>", placeholder)
    token1set, token2set = set(token1), set(token2)
    dif_token1, dif_token2 = token1set.difference(alphabet_set), token2set.difference(alphabet_set)

    if token1set.issubset(bigletter_set) and token2set.issubset(bigletter_set):
        return True
    if not token1set.issubset(alphabet_set) and not token2set.issubset(alphabet_set):
        if len(dif_token1) == 1 and len(dif_token2) == 1:
            if "." in dif_token1 and "." in dif_token2:
                return False
            if "_" in dif_token1 and "_" in dif_token2:
                return False
            if colon_token(token1) and colon_token(token2) and ":" in dif_token1 and ":" in dif_token2:
                return False
        if len(dif_token1) == 2 and len(dif_token2) == 2:
            if func_token(token1) and func_token(token2):
                return False
        if "....." in token1 or "....." in token2:
            return False
        return True
    if parenthesis_match(token1, token2set, placeholder) or parenthesis_match(token2, token1set, placeholder):
        return True

    return False


def greedy_merge_two_vars(str1, str2):
    """
    Greedily merge two variable strings from both sides.

    :param str1: The first variable string.
    :param str2: The second variable string.
    :return: The merged string.
    """
    if str1 == "<*>" or str2 == "<*>":
        return "<*>"
    placeholder = "\0"
    str1 = str1.replace("<*>", placeholder)
    str2 = str2.replace("<*>", placeholder)
    if not judge_var_token(str1, str2, placeholder):
        return False
    min_length = min(len(str1), len(str2))
    result = []

    for i in range(min_length):
        if str1[i] == str2[i]:
            result.append(str1[i])
        else:
            result.append(placeholder)

    if len(str1) > min_length:
        result.extend(str1[min_length:])
    elif len(str2) > min_length:
        result.extend(str2[min_length:])

    merged_string = ''.join(result)
    while placeholder+placeholder in merged_string:
        merged_string = merged_string.replace(placeholder+placeholder, placeholder)
    merged_string = merged_string.replace(placeholder, "<*>")

    return merged_string


def greedy_merge_two_vars_both_side(str1, str2):
    placeholder = "<*>"
    if str1 == "<*>" or str2 == "<*>":
        return "<*>"
    if not judge_var_token(str1, str2):
        return False

    i = 0
    while i < len(str1) and i < len(str2) and str1[i] == str2[i]:
        i += 1

    j = 0
    while j < len(str1) - i and j < len(str2) - i and str1[-(j + 1)] == str2[-(j + 1)]:
        j += 1

    result = []
    if i > 0:
        result.append(str1[:i])
    result.append(placeholder)
    if j > 0:
        result.append(str1[-j:])
    merged_string = ''.join(result)
    while placeholder+placeholder in merged_string:
        merged_string = merged_string.replace(placeholder+placeholder, placeholder)

    return merged_string


def greedy_merge_two_lists_both_side(list1, list2):
    """
    Greedily merge two lists from both sides.

    :param list1: The first list.
    :param list2: The second list.
    :return: The merged list.
    """
    placeholder = "<*>"
    if list1.count(placeholder) <= 1 and list2.count(placeholder) <= 1:
        return False
    if list1 == [placeholder] or list2 == [placeholder]:
        return [placeholder]

    len1, len2 = len(list1), len(list2)
    i = 0
    while i < len1 and i < len2 and (list1[i] == list2[i] or list1[i] == placeholder or list2[i] == placeholder):
        if list1[i] == placeholder or list2[i] == placeholder:
            break
        i += 1

    # 后向匹配
    j = 0
    while j < len1 - i and j < len2 - i and (
            list1[-(j + 1)] == list2[-(j + 1)] or list1[-(j + 1)] == placeholder or list2[-(j + 1)] == placeholder):
        if list1[-(j + 1)] == placeholder or list2[-(j + 1)] == placeholder:
            break
        j += 1

    result = []
    if i > 0:
        result.extend(list1[:i])
    result.append(placeholder)
    if j > 0:
        result.extend(list1[-j:])
    merged_result = []
    skip = False
    for k in range(len(result)):
        if result[k] == placeholder:
            if not skip:
                merged_result.append(placeholder)
                skip = True
        else:
            merged_result.append(result[k])
            skip = False

    return merged_result


def merge_sorted_lists(list1, list2):
    """
    Merge two sorted lists.

    :param list1: The first sorted list.
    :param list2: The second sorted list.
    :return: The merged sorted list.
    """
    merged_list = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            if not merged_list or merged_list[-1] != list1[i]:
                merged_list.append(list1[i])
            i += 1
        elif list1[i] > list2[j]:
            if not merged_list or merged_list[-1] != list2[j]:
                merged_list.append(list2[j])
            j += 1
        else:
            if not merged_list or merged_list[-1] != list1[i]:
                merged_list.append(list1[i])
            i += 1
            j += 1

    while i < len(list1):
        if not merged_list or merged_list[-1] != list1[i]:
            merged_list.append(list1[i])
        i += 1

    while j < len(list2):
        if not merged_list or merged_list[-1] != list2[j]:
            merged_list.append(list2[j])
        j += 1

    return merged_list


class TemplateDatabase:
    """
    A class for managing a database of log templates.

    Attributes:
        None (as the __init__ method is currently empty).
    """

    def __init__(self):
        self.template_items = {}
        self.template_list = []
        self.template_token_list = []
        self.sim_jaccard_threshold = 0.8
        self.norm_sim_edit_threshold = 0.3

    def add_template(self, event_template, indexes={}, relevant_templates=[]):
        """
        Add a new template to the database.

        :param event_template: The log template to be added.
        :param indexes: A dictionary of indexes related to the template. Defaults to {}.
        :param relevant_templates: A list of relevant templates. Defaults to [].
        """
        template_tokens = split_template_naive(event_template)
        if not template_tokens or event_template == "<*>":
            return False, event_template, None
        if len(self.template_items) == 0:
            self.insert_template(event_template, indexes)
            return False, event_template, None
        if len(template_tokens) == 1:
            self.insert_template(event_template, indexes)
            return False, event_template, None

        self.template_token_list = [split_template_naive(t) for t in self.template_list]
        coarse_similarities = [jaccard_similarity(template_tokens, t) for t in self.template_token_list]

        # only compare with the most similar template
        max_sim_idx = coarse_similarities.index(max(coarse_similarities))
        if self.judge_template_merge_combine(event_template, self.template_list[max_sim_idx]):
            print(f"[TemplateDB] Try Merge: `{event_template}` | `{self.template_list[max_sim_idx]}`")
            new_template, flag_merge_success = merge_template_by_star(event_template, self.template_list[max_sim_idx])
            if flag_merge_success:
                insert_indexes = self.update_template(new_template, indexes, max_sim_idx)
                self.template_items[new_template]['ori_templates'].append(event_template)
                print(f"[TemplateDB] Merged: -> `{new_template}`")
                return True, new_template, insert_indexes
            else:
                self.insert_template(event_template, indexes)
                print(f"[TemplateDB] Reject Merge, Remain Template: `{event_template}`")
                return False, event_template, None
        elif self.judge_template_merge_trace_attribute(event_template, self.template_list[max_sim_idx]):
            pass
        else:
            self.insert_template(event_template, indexes)
            return False, event_template, None

    def judge_template_merge_naive(self, template1, template2, split=[" "]):
        """
        Judge if two templates can be merged using a naive method.

        :param template1: The first template.
        :param template2: The second template.
        :param split: A list of characters to split the templates by. Defaults to [" "].
        :return: True if the templates can be merged, False otherwise.
        """
        parts1 = split_template(template1, split)
        parts2 = split_template(template2, split)
        sim_edit = edit_distance(parts1, parts2)
        sim_jaccard = jaccard_similarity(parts1, parts2)
        if (sim_edit <= 1) or (sim_jaccard > self.sim_jaccard_threshold):
            return True
        len1, len2 = len(parts1), len(parts2)
        if len1 > 10 and len2 > 10:  # assuming sequences with length > 10 are long sequences
            norm_sim_edit = sim_edit / max(len1, len2)  # normalized edit distance
            if norm_sim_edit <= self.norm_sim_edit_threshold:  # assuming norm_sim_edit_threshold is a predefined threshold
                return True
        return False

    def judge_template_merge_simple(self, template1, template2, split=[" "]):
        """
        Judge if two templates can be merged using a simple method.

        :param template1: The first template.
        :param template2: The second template.
        :param split: A list of characters to split the templates by. Defaults to [" "].
        :return: True if the templates can be merged, False otherwise.
        """
        parts1 = split_template(template1, split)
        parts2 = split_template(template2, split)
        if len(parts1) != len(parts2):
            return False
        edit_num = sum([p1!=p2 for p1, p2 in zip(parts1, parts2)])
        if edit_num/len(parts1) < 0.2:
            return True
        # if edit_num <= 1:
        #     return True
        return False

    def judge_template_merge_combine(self, template1, template2, split=[" "]):
        """
        Judge if two templates can be merged using a combined method.

        :param template1: The first template.
        :param template2: The second template.
        :param split: A list of characters to split the templates by. Defaults to [" "].
        :return: True if the templates can be merged, False otherwise.
        """
        parts1 = split_template(template1, split)
        parts2 = split_template(template2, split)
        if len(parts1) != len(parts2):
            return False
        edit_num = sum([p1!=p2 for p1, p2 in zip(parts1, parts2)])
        if edit_num <= 1:
            return True
        elif edit_num == 2 and len(parts1) > 10:
            return True
        return False

    def judge_template_merge_trace_attribute(self, template1, template2, split=[" "]):
        """
        Judge if two templates can be merged based on trace attributes.

        :param template1: The first template.
        :param template2: The second template.
        :param split: A list of characters to split the templates by. Defaults to [" "].
        :return: True if the templates can be merged, False otherwise.
        """
        parts1 = split_template(template1, split)
        parts2 = split_template(template2, split)

    def insert_template(self, event_template, indexes):
        template_tokens = split_template_naive(event_template)
        self.template_items[event_template] = {'len': len(template_tokens), 'indexes': indexes, 'ori_templates': [event_template]}
        self.template_list.append(event_template)
        self.template_token_list.append(template_tokens)

    def update_template(self, new_template, new_indexes, idx):
        old_template = self.template_list[idx]
        template_tokens = split_template_naive(new_template)

        insert_indexes = self.template_items[old_template].get('indexes', {}).copy()
        for k, v in new_indexes.items():
            if k in insert_indexes:
                insert_indexes[k] = merge_sorted_lists(v, insert_indexes[k])
            else:
                insert_indexes[k] = v
        self.template_items[new_template] = {'len': len(template_tokens), 'indexes': insert_indexes,
                                             'ori_templates': self.template_items[old_template]['ori_templates']}
        if new_template != old_template:
            self.template_items.pop(old_template)
            self.template_list.pop(idx)
            self.template_token_list.pop(idx)
            self.template_list.append(new_template)
            self.template_token_list.append(split_template_naive(new_template))
        return insert_indexes

    def update_indexes(self, template, new_indexes):
        """
        Update the indexes of an existing template in the database.

        :param template: The log template whose indexes need to be updated.
        :param new_indexes: A dictionary of new indexes.
        """
        # old_template = self.template_list[idx]
        if template not in self.template_items:
            template_tokens = split_template_naive(template)
            self.template_items[template] = {'len': len(template_tokens), 'indexes': new_indexes, 'ori_templates': [template]}
            self.template_list.append(template)
            self.template_token_list.append(template_tokens)
            return new_indexes
        else:
            indexes2 = self.template_items[template].get('indexes', {}).copy()
            for k, v in new_indexes.items():
                if k in indexes2:
                    indexes2[k] = merge_sorted_lists(v, indexes2[k])
                else:
                    indexes2[k] = v
            print(f"[TemplateDB] Update Indexes: {sum(len(v) for v in self.template_items[template].get('indexes', {}).values())} -> {sum(len(v) for v in indexes2.values())} for `{template}`")
            self.template_items[template]['indexes'] = indexes2
            self.template_items[template]['ori_templates'].append(template)
            return indexes2

    def print_database(self):
        print("========= Template Database =========")
        for template in self.template_list:
            print(f"\t{template} || {self.template_items[template]['indexes']}")
            print(f"\t\t{self.template_items[template]['ori_templates']}")

    def get_all_templates(self):
        return self.template_list


def post_process_tokens(tokens, punc):
    excluded_str = ['=', '|', '(', ')']
    for i in range(len(tokens)):
        if tokens[i].find("<*>") != -1:
            tokens[i] = "<*>"
        else:
            new_str = ""
            for s in tokens[i]:
                if (s not in punc and s != ' ') or s in excluded_str:
                    new_str += s
            tokens[i] = new_str
    return tokens


def message_split(message):
    """
    Split a message into parts.

    :param message: The message to be split.
    :return: The split parts of the message.
    """
    punc = "!\"#$%&'()+,-/:;=?@.[\]^_`{|}~"
    splitters = "\s\\" + "\\".join(punc)
    # print(splitters)
    # splitters = "\\".join(punc)
    # splitter_regex = re.compile("([{}]+)".format(splitters))
    splitter_regex = re.compile("([{}])".format(splitters))
    tokens = re.split(splitter_regex, message)

    tokens = list(filter(lambda x: x != "", tokens))

    # print("tokens: ", tokens)
    tokens = post_process_tokens(tokens, punc)

    tokens = [
        token.strip()
        for token in tokens
        if token != "" and token != ' '
    ]
    tokens = [
        token
        for idx, token in enumerate(tokens)
        if not (token == "<*>" and idx > 0 and tokens[idx - 1] == "<*>")
    ]
    return tokens


def tree_match(match_tree, log_content):
    """
    Match a log content against a match tree.

    :param match_tree: The match tree to be used for matching.
    :param log_content: The log content to be matched.
    :return: The result of the match.
    """

    log_tokens = message_split(log_content)
        #print("log tokens: ", log_tokens)
    template, template_id, parameter_str = match_template(match_tree, log_tokens)
    if template:
        return (template, template_id, parameter_str)
    else:
        return ("NoMatch", "NoMatch", parameter_str)


def match_template(match_tree, log_tokens):
    """
    Match a list of log tokens against a match tree.

    :param match_tree: The match tree to be used for matching.
    :param log_tokens: A list of log tokens to be matched.
    :return: The result of the match.
    """
    results = []
    find_results = find_template(match_tree, log_tokens, results, [], 1)
    relevant_templates = find_results[1]
    if len(results) > 1:
        new_results = []
        for result in results:
            if result[0] is not None and result[1] is not None and result[2] is not None:
                new_results.append(result)
    else:
        new_results = results
    if len(new_results) > 0:
        if len(new_results) > 1:
            new_results.sort(key=lambda x: (-x[1][0], x[1][1]))
        return new_results[0][1][2], new_results[0][1][3], new_results[0][2]
    return False, False, relevant_templates


def get_all_templates(move_tree):
    """
    Get all templates from a move tree.

    :param move_tree: The move tree to extract templates from.
    :return: A list of all templates in the move tree.
    """
    result = []
    for key, value in move_tree.items():
        if isinstance(value, tuple):
            result.append(value[2])
        else:
            result = result + get_all_templates(value)
    return result


def find_template(move_tree, log_tokens, result, parameter_list, depth):
    """
    Find a template in a move tree based on a list of log tokens.

    :param move_tree: The move tree to search in.
    :param log_tokens: A list of log tokens to match.
    :param result: The result object to store the matching information.
    :param parameter_list: A list to store the extracted parameters.
    :param depth: The current depth in the move tree.
    :return: The result of the template search.
    """
    flag = 0  # no further find
    if len(log_tokens) == 0:
        for key, value in move_tree.items():
            if isinstance(value, tuple):
                result.append((key, value, tuple(parameter_list)))
                flag = 2  # match
        if "<*>" in move_tree:
            parameter_list.append("")
            move_tree = move_tree["<*>"]
            if isinstance(move_tree, tuple):
                result.append(("<*>", None, None))
                flag = 2  # match
            else:
                for key, value in move_tree.items():
                    if isinstance(value, tuple):
                        result.append((key, value, tuple(parameter_list)))
                        flag = 2  # match
        # return (True, [])
    else:
        token = log_tokens[0]

        relevant_templates = []

        if token in move_tree:
            find_result = find_template(move_tree[token], log_tokens[1:], result, parameter_list, depth + 1)
            if find_result[0]:
                flag = 2  # match
            elif flag != 2:
                flag = 1  # further find but no match
                relevant_templates = relevant_templates + find_result[1]
        if "<*>" in move_tree:
            if isinstance(move_tree["<*>"], dict):
                next_keys = move_tree["<*>"].keys()
                next_continue_keys = []
                for nk in next_keys:
                    nv = move_tree["<*>"][nk]
                    if not isinstance(nv, tuple):
                        next_continue_keys.append(nk)
                idx = 0
                # print("len : ", len(log_tokens))
                while idx < len(log_tokens):
                    token = log_tokens[idx]
                    # print("try", token)
                    if token in next_continue_keys:
                        # print("add", "".join(log_tokens[0:idx]))
                        parameter_list.append("".join(log_tokens[0:idx]))
                        # print("End at", idx, parameter_list)
                        find_result = find_template(
                            move_tree["<*>"], log_tokens[idx:], result, parameter_list, depth + 1
                        )
                        if find_result[0]:
                            flag = 2  # match
                        elif flag != 2:
                            flag = 1  # further find but no match
                        if parameter_list:
                            parameter_list.pop()
                    idx += 1
                if idx == len(log_tokens):
                    parameter_list.append("".join(log_tokens[0:idx]))
                    find_result = find_template(
                        move_tree["<*>"], log_tokens[idx + 1:], result, parameter_list, depth + 1
                    )
                    if find_result[0]:
                        flag = 2  # match
                    else:
                        if flag != 2:
                            flag = 1
                        relevant_templates = relevant_templates + find_result[1]
                    if parameter_list:
                        parameter_list.pop()
    if flag == 2:
        # Match
        return True, []
    if flag == 1:
        # Further find but no match
        return False, relevant_templates
    if flag == 0:
        # Not Find
        # print(log_tokens, flag)
        if depth >= 2:
            return False, get_all_templates(move_tree)
        else:
            return False, []
