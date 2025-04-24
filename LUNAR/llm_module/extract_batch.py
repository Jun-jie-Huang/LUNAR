import re
from typing import List, Any


class Extract(object):
    @staticmethod
    def extract(raw_response: str, **kwargs: Any) -> str:
        """
        Extract the answer from the raw response.
        Args:
            raw_response: raw response from model
            **kwargs: other arguments
        Returns: extracted result
        """
        return raw_response.strip()


class BatchExtract(Extract):
    @staticmethod
    def extract(raw_response: str, **kwargs: Any) -> List[Any]:
        """
        Extract the batch of multi-choice answers from raw_response by regex.

        Args:
            raw_response: raw response from model
            **kwargs: other arguments
        Returns: extracted result list

        """
        batch_answers = []
        for answer in raw_response.split("\n"):
            # Skip answer prefix
            answer = answer.strip()
            numbers = re.findall(r'\[(\d+)\]', answer)
            if len(numbers) > 0:
                idx = int(numbers[0])
            else:
                idx = -1
            if answer.startswith("LogTemplate:"):
                answer = answer[len("LogTemplate: "):]
            elif answer.startswith(f"LogTemplate[{idx}]: "):
                answer = answer[len(f"LogTemplate[{idx}]: "):]
            elif answer.startswith("Log template: "):
                answer = answer[len("Log template: "):]
            elif answer.startswith(f"Log template[{idx}]: "):
                answer = answer[len(f"Log template[{idx}]: "):]
            else:
                continue

            answer = answer.lstrip('`').rstrip('`')
            batch_answers.append({'idx': idx, 'template': answer})

        return batch_answers
