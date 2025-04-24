
import json
def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=1)
def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def get_max_retry(logs=[], max_retry_assigned=3):
    if logs == "":
        return max_retry_assigned
    if len(logs) == 0:
        return max_retry_assigned
    logs_length = [len(log.split(' ')) for log in logs]
    if len(logs) == 1 and logs_length[0] <= 5:
        return 1
    if all([i <= 2 for i in logs_length]):
        return 1
    return max_retry_assigned


import regex as re


def verify_template_for_log_regex(log, template):
    """
    input a log and a template, return True if the template matches the log, otherwise return False
    :param log:
    :param template:
    :return:
    """
    if "<*>" not in template:
        return log == template
    template_regex = re.sub(r"<.{1,5}>", "<*>", template)
    if "<*>" not in template_regex:
        return False
    template_regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template_regex)
    template_regex = re.sub(r"\\ +", r"\\s+", template_regex)
    template_regex = "^" + template_regex.replace("\<\*\>", "(?:.*?)") + "$"
    match = re.match(template_regex, log)
    return match is not None


def verify_template_for_log_with_first_token(log, template):
    """ Not always True, just test speed """
    tok_log, tok_temp = log.split(), template.split()
    if "<*>" not in tok_temp[0]  and tok_log[0] != tok_temp[0]:
        return False
    if "<*>" not in tok_temp[-1] and tok_log[-1] != tok_temp[-1]:
        return False

    return verify_template_for_log_regex(log, template)


def verify_template_for_log_with_first_token_subset(log, template):
    """ Not always True, just test speed """
    tok_log, tok_temp = log.split(), template.split()
    if "<*>" not in tok_temp[0] and tok_log[0] != tok_temp[0]:
        return False
    if "<*>" not in tok_temp[-1] and tok_log[-1] != tok_temp[-1]:
        return False
    set_log = set(tok_log)
    set_temp = set(i for i in tok_temp if "<*>" not in i)
    if not set_temp.issubset(set_log):
        return False

    return verify_template_for_log_regex(log, template)

def verify_template_and_update(row, new_template):
    is_valid = verify_template_for_log_with_first_token(row["Content"], template=new_template)
    if is_valid:
        row["Template"] = new_template
    return row, is_valid


def get_parameter_list(log, template):
    template_regex = re.sub(r"<.{1,5}>", "<*>", template)
    if "<*>" not in template_regex:
        return []
    template_regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template_regex)
    template_regex = re.sub(r"\\ +", r"\\s+", template_regex)
    template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
    parameter_list = re.findall(template_regex, log)
    parameter_list = parameter_list[0] if parameter_list else ()
    parameter_list = (
        list(parameter_list)
        if isinstance(parameter_list, tuple)
        else [parameter_list]
    )
    return parameter_list


def validate_template(template):
    if len(template) == 0:
        return False
    if template.count("<*>") > 50:
        return False

    return True


def preprocess_log_for_query(log, regexes):
    for currentRex in regexes:
        log = re.sub(currentRex, "<*>", log)
    return log
