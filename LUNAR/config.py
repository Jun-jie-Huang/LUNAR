import argparse
import os


datasets_all = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
    "LoFI",
    "CTS",
    "HiBench",
]


datasets_2k = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
    "LoFI",
    "CTS",
    "HiBench",
]

datasets_full = [
    "Proxifier",
    "Apache",
    "OpenSSH",
    "HDFS",
    "OpenStack",
    "HPC",
    "Zookeeper",
    "HealthApp",
    "Hadoop",
    "Spark",
    "BGL",
    "Linux",
    "Mac",
    "Thunderbird",
    "LoFI",
    "CTS",
    "HiBench",
]

benchmark_settings = {
    "HDFS": {
        "log_file": "HDFS/HDFS_full.log",
        "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "regex": [r"/?(\d+\.){3}\d+(:\d+)?"],
        'delimiter': [''],
        "st": 0.5,
        "depth": 4,
    },
    "Hadoop": {
        "log_file": "Hadoop/Hadoop_full.log",
        "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
        "regex": [],
        'delimiter': [],
        "st": 0.5,
        "depth": 4,
    },
    "Spark": {
        "log_file": "Spark/Spark_full.log",
        "log_format": "<Date> <Time> <Level> <Component>: <Content>",
        "regex": [],
        'delimiter': [],
        "st": 0.5,
        "depth": 4,
    },
    "Zookeeper": {
        "log_file": "Zookeeper/Zookeeper_full.log",
        "log_format": "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
        "regex": [r"(/|)(\d+\.){3}\d+(:\d+)?"],
        'delimiter': [],
        "st": 0.5,
        "depth": 4,
    },
    "BGL": {
        "log_file": "BGL/BGL_full.log",
        "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        "regex": [],
        'delimiter': [],
        "st": 0.5,
        "depth": 4,
    },
    "HPC": {
        "log_file": "HPC/HPC_full.log",
        "log_format": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
        "regex": [r"node-D?\d+\\?", r"node-D?\[.*?\]"],
        'delimiter': [],
        "st": 0.5,
        "depth": 4,
    },
    "Thunderbird": {
        "log_file": "Thunderbird/Thunderbird_full.log",
        "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>",
        "regex": [],
        'delimiter': [],
        "st": 0.5,
        "depth": 4,
    },
    "Linux": {
        "log_file": "Linux/Linux_full.log",
        "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
        'regex': [r'J([a-z]{2})'],
        'delimiter': [r''],
        "st": 0.39,
        "depth": 6,
    },
    "HealthApp": {
        "log_file": "HealthApp/HealthApp_full.log",
        "log_format": "<Time>\|<Component>\|<Pid>\|<Content>",
        "regex": [],
        'delimiter': [r''],
        "st": 0.2,
        "depth": 4,
    },
    "Apache": {
        "log_file": "Apache/Apache_full.log",
        "log_format": "\[<Time>\] \[<Level>\] <Content>",
        "regex": [r'HTTP/\d\.\d'],
        'delimiter': [],
        "st": 0.5,
        "depth": 4,
    },
    "Proxifier": {
        "log_file": "Proxifier/Proxifier_full.log",
        "log_format": "\[<Time>\] <Program> - <Content>",
        "regex": [
            r"<\d+\ssec",
            r"([\w-]+\.)+[\w-]+(:\d+)?",
            r"[KGTM]B",
        ],
        'delimiter': [r'\(.*?\)'],
        "st": 0.6,
        "depth": 3,
    },
    "OpenSSH": {
        "log_file": "OpenSSH/OpenSSH_full.log",
        "log_format": "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
        "regex": [r"([\w-]+\.){2,}[\w-]+", r'HTTP/\d\.\d'],
        'delimiter': [],
        "st": 0.6,
        "depth": 5,
    },
    "OpenStack": {
        "log_file": "OpenStack/OpenStack_full.log",
        "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
        "regex": [r'HTTP/\d\.\d'],
        'delimiter': [],
        "st": 0.5,
        "depth": 5,
    },
    "Mac": {
        "log_file": "Mac/Mac_full.log",
        "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
        "regex": [r"([\w-]+\.){2,}[\w-]+"],
        'delimiter': [],
        "st": 0.7,
        "depth": 6,
    },
    "LoFI": {
        "log_file": "Industry-clean/Industry-clean_full.log",
        "log_format": "<Content>",
        "regex": [r"(?<=request-id:)routeid(?=,)"],
        'delimiter': [],
        "st": 0.7,
        "depth": 6,
    },
    "CTS": {
        "log_file": "CTS/CTS_full.log",
        "log_format": "\[<ID>\]\[<Process>\] <Content>",
        "regex": [r'\d+/\d+/\d+$'],
        'delimiter': [],
        "st": 0.7,
        "depth": 6,
    },
    "HiBench": {
        "log_file": "HiBench/HiBench_full.log",
        "log_format": "\[<Time>\] <Content>",
        "regex": [r"hdfs://[a-zA-Z0-9._-]+(/[a-zA-Z0-9._/-]*)*", r"^/[a-zA-Z0-9._-]+(/[a-zA-Z0-9._-]*)*"],
        'delimiter': [],
        "st": 0.7,
        "depth": 6,
    },
}


LLM_BASE_MAPPING = {
    "gpt35-0125": ["gpt-3.5-turbo-0125", "API_BASE_URL", "API_KEY"],
}


def common_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_id', type=str, default="00")
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--base_dir', type=str, default="./")
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--gt_dir', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="")

    parser.add_argument('--data_type', type=str, default="full", choices=["full", "2k"], help="Set this if you want to test on full dataset")
    parser.add_argument('--test_dataset', type=str, default="Apache", choices=datasets_all)
    parser.add_argument('--otc',
                        help="Set this if you want to use corrected oracle templates",
                        default=False, action='store_true')
    parser.add_argument('--complex', type=int,
                        help="Set this if you want to test on complex dataset",
                        default=0)
    parser.add_argument('--frequent', type=int,
                        help="Set this if you want to test on frequent dataset",
                        default=0)
    parser.add_argument("--verbose", action="store_true", default=False)

    parser.add_argument('--add_regex', type=str, default="before", choices=["add", "no", "before"])
    parser.add_argument('--regex', type=str, default="")
    parser.add_argument('--llm', type=str, default="gpt35-0125")
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--base_url', type=str, default="")
    parser.add_argument('--api_key', type=str, default="")
    parser.add_argument('--llm_params', type=dict, default={})
    parser.add_argument('--cluster_params', type=dict, default={})

    parser.add_argument("--parallel", action="store_true", default=False)
    parser.add_argument('--thread', type=int, default=8)

    return parser


def parameter_args(parser):
    parser.add_argument('--max_retry', type=int, default=2)
    parser.add_argument('--shot', type=int, default=0)

    parser.add_argument('--cluster_method', type=str, default="TopKToken", choices=["TopKToken"])
    parser.add_argument('--cluster_topk', type=int, default=3)
    parser.add_argument('--min_cluster_size', type=int, default=100)

    parser.add_argument('--sample_method', type=str, default="lcu_sampling", choices=["lcu_sampling"])
    parser.add_argument('--sample_size', type=int, default=3, help="parameter 1")
    parser.add_argument('--sample_min_similarity', type=float, default=0.5, help="parameter 2")
    parser.add_argument('--lcu_lamb', type=float, default=0.6, help="parameter 3")
    parser.add_argument('--lcu_sample_size', type=int, default=3, help="parameter 4")
    parser.add_argument('--sample_size_auto', type=str, default="auto", choices=["fixed", "auto"])
    parser.add_argument("--add_skip_sim", action="store_true", default=False)
    parser.add_argument("--not_pad_query", action="store_true", default=True)
    parser.add_argument('--template_refine', type=str, default="TempDB", choices=["No", "TempDB"])

    parser.add_argument('--prompt', type=str, default="VarExam", choices=["VarOnly", "VarExam", "NoPE", "NoOutputConstraint", "NoAdvice", "SimpleRequirements"])

    return parser


def load_args():
    parser = common_args()
    parser = parameter_args(parser)
    args = parser.parse_args()
    args.parallel = False

    args.model = LLM_BASE_MAPPING[args.llm][0] if args.model == "" else args.model
    args.base_url = LLM_BASE_MAPPING[args.llm][1] if args.base_url == "" else args.base_url
    args.api_key = LLM_BASE_MAPPING[args.llm][2] if args.api_key == "" else args.api_key
    args.llm_params = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "dataset": args.test_dataset,
        "prompt": args.prompt,
    }
    args.cluster_params = {
        "cluster_method": args.cluster_method,
        "cluster_topk": args.cluster_topk,
        "min_cluster_size": args.min_cluster_size,
        "sample_method": args.sample_method,
        "sample_size": args.sample_size,
        "sample_min_similarity": args.sample_min_similarity,
        "add_skip_sim": args.add_skip_sim,
        "pad_query": not args.not_pad_query,
        "lcu_lamb": args.lcu_lamb,
        "lcu_sample_size": args.lcu_sample_size,
        "sample_size_auto": args.sample_size_auto,
        "add_regex": args.add_regex,
        "regex": [],
        "sample_size_assigned": args.sample_size,
        "dedup": True,
    }

    print(f"[PARAM]: pad_query: {args.cluster_params['pad_query']}")
    print(f"[PARAM]: add_skip_sim: {args.cluster_params['add_skip_sim']}")
    print(f"[PARAM]: cluster_method: {args.cluster_params['cluster_method']}")
    print(f"[PARAM]: cluster_topk: {args.cluster_params['cluster_topk']}")
    print(f"[PARAM]: sample_method: {args.cluster_params['sample_method']}")
    print(f"[PARAM]: lcu_sample_size: {args.cluster_params['lcu_sample_size']}")
    print(f"[PARAM]: lcu_lamb: {args.cluster_params['lcu_lamb']}")

    # input dir
    args.data_dir = os.path.join(args.base_dir, "./",  "datasets")
    if args.gt_dir == "":
        args.gt_dir = args.data_dir

    # output dir
    args.prefix = "LUNAR-single"
    args.output_dir = os.path.join(args.base_dir, "saved_results", args.prefix)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"Save dir: {args.output_dir}")

    if args.data_type == "full":
        args.otc = False
    args.verbose = True
    if args.add_regex == 'add':
        args.regex = benchmark_settings[args.test_dataset]["regex"]
    if args.add_regex == 'before':
        args.regex = benchmark_settings[args.test_dataset]["regex"]
        args.cluster_params['regex'] = args.regex
    else:
        args.regex = []

    return args


def load_args_parallel():
    parser = common_args()
    parser = parameter_args(parser)
    args = parser.parse_args()
    args.parallel = True

    args.model = LLM_BASE_MAPPING[args.llm][0] if args.model == "" else args.model
    args.base_url = LLM_BASE_MAPPING[args.llm][1] if args.base_url == "" else args.base_url
    args.api_key = LLM_BASE_MAPPING[args.llm][2] if args.api_key == "" else args.api_key
    args.llm_params = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "dataset": args.test_dataset,
        "prompt": args.prompt,
    }
    args.cluster_params = {
        "cluster_method": args.cluster_method,
        "cluster_topk": args.cluster_topk,
        "min_cluster_size": args.min_cluster_size,
        "sample_method": args.sample_method,
        "sample_size": args.sample_size,
        "sample_min_similarity": args.sample_min_similarity,
        "add_skip_sim": args.add_skip_sim,
        "pad_query": not args.not_pad_query,
        "lcu_lamb": args.lcu_lamb,
        "lcu_sample_size": args.lcu_sample_size,
        "sample_size_auto": args.sample_size_auto,
        "add_regex": args.add_regex,
        "regex": [],
        "sample_size_assigned": args.sample_size,
        "dedup": True,
    }

    print(f"[PARAM]: pad_query: {args.cluster_params['pad_query']}")
    print(f"[PARAM]: add_skip_sim: {args.cluster_params['add_skip_sim']}")
    print(f"[PARAM]: cluster_method: {args.cluster_params['cluster_method']}")
    print(f"[PARAM]: cluster_topk: {args.cluster_params['cluster_topk']}")
    print(f"[PARAM]: sample_method: {args.cluster_params['sample_method']}")
    print(f"[PARAM]: lcu_sample_size: {args.cluster_params['lcu_sample_size']}")
    print(f"[PARAM]: lcu_lamb: {args.cluster_params['lcu_lamb']}")

    # input dir
    if args.data_type == "full":
        args.data_dir = os.path.join(args.base_dir, "./",  "datasets")
        # args.data_dir = os.path.join(args.base_dir, "./",  "loghub2_correct")
    else:
        args.data_dir = os.path.join(args.base_dir, "./",  "2k_dataset")
    if args.gt_dir == "":
        args.gt_dir = args.data_dir

    # # output dir
    args.prefix = "LUNAR-parallel"
    args.output_dir = os.path.join(args.base_dir, "saved_results", args.prefix)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"Save dir: {args.output_dir}")

    if args.data_type == "full":
        args.otc = False
    args.verbose = True
    if args.add_regex == 'add':
        args.regex = benchmark_settings[args.test_dataset]["regex"]
    if args.add_regex == 'before':
        args.regex = benchmark_settings[args.test_dataset]["regex"]
        args.cluster_params['regex'] = args.regex
    else:
        args.regex = []

    return args

