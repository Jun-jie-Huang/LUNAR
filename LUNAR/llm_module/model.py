import time
from openai import OpenAI
from LUNAR.llm_module.response_extractor.extract_batch import BatchExtract
from LUNAR.llm_module.post_process import post_process_template
from LUNAR.llm_module.template_aggregator import aggregate_by_majority
from LUNAR.llm_module.variable_examples import VARIABLE_EXAMPLES_SETTING, json2prompt


class InferLLMGrouping:
    def __init__(self, model, api_key, base_url, prefix=None, dataset="Apache", prompt="VarExam"):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.prefix = prefix
        self.dataset = dataset
        self.prompt = prompt

        self.module_params = {}
        self.messages = []
        self.usages = []
        self.response = ""

        self.system_prompt = ("You are a log parsing assistant for the cloud reliability team, "
                              "skilled in identifying dynamic values of variables/parameters in logs. "
                              "The value is an actual manifestation of the variables in original logging statements.\n"
                              )
        self.prompt_base_requirements = (
            "# Basic Requirements:\n"
            "- I will provide multiple log messages, each delimited by backticks.\n"
            "- You must identify and extract all dynamic variables in each log with {placeholder} and output static log templates.\n"
            "- Identify the semantics of variables and compare the differences between logs to identify potential dynamic variables if they belong to the same template.\n"
            "- Preserve any dynamic variables already marked by `<*>` or `{placeholder}`.\n"
            "- Pay attention to the slightly different strings among logs, which have high possibility to be dynamic variable.\n"
            "- Do not convert non-variables, especially when only one log is presented in the group.\n"
        )
        if "SimpleRequirements" in self.prompt:
            self.prompt_base_requirements = (
                "# Basic Requirements:\n"
                "- I want you to act like an expert in log parsing.\n"
                "- I will give you a log message wrapped by backticks.\n"
                "- Your task is to identify all the dynamic variables in logs, replace them with {variables}, and output a static log template.\n"
                "- Please print the input log's template wrapped by backticks.\n"
            )

        if "NoAdvice" not in self.prompt:
            self.prompt_variable_advice = (
                "# Advices on variables:\n"
                "- Common variables: numbers, IP addresses, URLs, file paths, directories, hex values, usernames, etc.\n"
                "- Full directory with filename, complex url with server address or domain should be recognize as one variable.\n"
                "# Advices on non-variables:\n"
                "- Error messages/types, java exceptions, detailed commands or interrupted messages are NOT dynamic variables as they contain important information.\n"
                "- Specific actions or status words are NOT dynamic variables.\n"
            )
        else:
            self.prompt_variable_advice = ""

        if "NoPE" not in self.prompt:
            self.prompt_variable_example_prompt = self.construct_variable_example()
            print(self.prompt_variable_example_prompt)
        else:
            self.prompt_variable_example_prompt = ""

        if "NoOutputConstraint" not in self.prompt:
            self.prompt_output_constraint = (
                "# Output Constraints: \n"
                "- For each log line, output corresponding log template starting with LogTemplate[idx], no other line break. \n"
                "- Each input log's template is delimited by backticks. \n"
            )
        else:
            self.prompt_output_constraint = ""

        self.instruction = ""
        self.instruction += self.prompt_base_requirements
        self.instruction += self.prompt_variable_advice
        self.instruction += self.prompt_variable_example_prompt
        self.instruction += self.prompt_output_constraint

        print("======================== Prompt ========================")
        print(self.prompt)
        print(self.instruction)
        print("======================== Prompt ========================")

    def construct_variable_example(self):
        pe_dict = VARIABLE_EXAMPLES_SETTING['lunar']['variable_examples']
        prompt = json2prompt(pe_dict)

        return prompt

    def get_prompt_direct(self, logs, exemplars=None):
        # print(instruction)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.instruction},
            {"role": "assistant", "content": "OK, I'm ready to help."},
        ]
        if exemplars is not None:
            examplar_logs = [exemplar['query'] for exemplar in exemplars]
            examplar_templates = [exemplar['answer'] for exemplar in exemplars]
            query_template = '\n'.join([f"Log[{i+1}]: " + "`{}`" for i in range(len(exemplars))])
            answer_template = '\n'.join([f"LogTemplate[{i+1}]: " + "`{}`" for i in range(len(exemplars))])
            messages.append({"role": "user", "content": query_template.format(*examplar_logs)})
            messages.append({"role": "assistant", "content": answer_template.format(*examplar_templates)})

        query_template = '\n'.join([f"Log[{i+1}]: "+"`{}`" for i in range(len(logs))])
        query = query_template.format(*logs)
        messages.append({"role": "user", "content": query})
        # self.messages = messages
        print("\t============  Query  ====================")
        print("\n".join(["\t"+i for i in query.split('\n')]))
        return messages

    def parsing_log_templates(self, logs, exemplars, gts=[], reparse=False):
        # query llm for response
        messages = self.get_prompt_direct(logs, exemplars=exemplars)
        temperature = 0.7 if reparse else 0.0
        time1 = time.time()
        _ = self.get_response_fallback(messages, temperature=temperature)
        query_time = time.time() - time1

        # print response
        print("\t============ Response ====================")
        print(self.response)
        if len(gts) > 0:
            print("\t============ Target ====================")
            answer_template = '\n'.join([f"\tGT Template[{i+1}]: " + "`{}`" for i in range(len(gts))])
            print(answer_template.format(*gts))
        # print("================================")

        # post process response
        try:
            gpt_templates = self.extract_and_post_process(logs, self.response)
            templates = [temp['post_process'] for temp in gpt_templates]
        except:
            templates = [post_process_template(log, [])[0] for log in logs]

        # aggregate templates
        best_template = aggregate_by_majority(logs, templates)

        return best_template, query_time

    def get_response(self, messages, temperature=0.0):
        answers = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            seed=1603,
            n=1,
            stop=None,
        )
        self.usages.append({"query": answers.usage.prompt_tokens, "response": answers.usage.completion_tokens})
        self.response = [response.message.content for response in answers.choices if response.finish_reason != 'length'][0]
        return self.response

    def get_response_fallback(self, messages, temperature=0.0):
        retry_times = 0
        while retry_times < 3:
            try:
                answers = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    seed=1603,
                    n=1,
                    stop=None,
                )
                self.response = [response.message.content for response in answers.choices][0]
                return self.response

            except Exception as e:
                print("Exception :", e)
                if "list index out of range" in str(e):
                    break
                retry_times += 1
        return ""

    def get_compromise_response(self, logs):
        return post_process_template(logs[0], [])[0]

    def extract_and_post_process(self, logs, response):
        gpt_templates = BatchExtract.extract(response)

        # replace null template with previous template
        gpt_templates = self.make_up_template(logs, gpt_templates)

        print("\t============ PostProcess ====================")
        for temp in gpt_templates:
            new_temp, _ = post_process_template(temp['template'], [])
            temp['post_process'] = new_temp
        return gpt_templates

    @staticmethod
    def make_up_template(logs, templates):
        """
            replace missing template with previous template
        :param logs: a list of strings
        :param templates: a list of dictionaries, each dictionary contains 'idx' and 'template'
        :return:
        """
        templates = sorted(templates, key=lambda x: x['idx'])
        # remove null template
        templates = [d for d in templates if d.get('idx') != -1]
        if len(templates) == 0:
            return [{'idx': i+1, 'template': log} for i, log in enumerate(logs)]

        new_templates = []
        existing_idx = [d['idx'] for d in templates]
        # print(existing_idx)
        template_idx = -1
        for idx, _log in enumerate(logs):
            if idx + 1 not in existing_idx:
                new_templates.append({'idx': idx+1, 'template': templates[0]['template']})
            else:
                template_idx += 1
                new_templates.append({'idx': idx+1, 'template': templates[template_idx]['template']})
        new_templates = sorted(new_templates, key=lambda x: x['idx'])

        return new_templates

