from collections import Counter
from ..utils import validate_template

def aggregate_by_first(logs, templates=[]):
    return templates[0]
    

def aggregate_by_majority(logs, templates=[]):
    templates = [t for t in templates if validate_template(t)]
    if len(templates) == 0:
        return ""
    else:
        counter = Counter(templates)
        mode_template = counter.most_common(1)[0][0]
        return mode_template


def aggregate_by_llm(logs, templates=[]):
    pass


def possible_requery(logs, templates):
    # if we have diverse templates (num/logs > 80%), we need to requery
    if len(logs) <= 1:
        return False
    counter = Counter(templates)
    if len(counter) / len(logs) >= 0.8:
        return True
    return False


