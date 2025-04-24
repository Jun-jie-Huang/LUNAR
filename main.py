import os
import time
import random
from LUNAR.LUNAR import LUNARParser
from LUNAR.config import load_args
from utils.evaluator_main import evaluator, prepare_results, post_average

random.seed(22222)
if __name__ == "__main__":

    random.seed(22222)
    print("Start to parse logs")
    args = load_args()

    input_dataset_dir = os.path.join(args.data_dir, args.test_dataset)
    output_dataset_dir = os.path.join(args.output_dir, args.test_dataset)
    print(f"Input dir: {input_dataset_dir}")
    print(f"Output dir: {output_dataset_dir}")
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)

    time1 = time.time()
    parser = LUNARParser(add_regex=args.add_regex, regex=args.regex, data_type=args.data_type,
                                            dir_in=input_dataset_dir, dir_out=output_dataset_dir, rex=[],
                                          llm_params=args.llm_params, shot=args.shot,
                                          cluster_params=args.cluster_params,
                                          )
    parser.parse(args.test_dataset)
    total_time = time.time() - time1
    print(f"Total parsingg time: {total_time} seconds")
    print(f"Total query time: {parser.wait_query_time} seconds, {parser.query_count} queries")
    print(f"Avg. query time: {parser.wait_query_time/(parser.query_count+0.00001)} seconds")
    print(f"Total pure match time: {total_time - parser.wait_query_time} seconds")

    result_file = prepare_results(args.output_dir, otc=args.otc, complex=args.complex, frequent=args.frequent)
    evaluator(args.test_dataset, args.data_type, input_dataset_dir, output_dataset_dir, result_file,
              otc=args.otc, complex=args.complex, frequent=args.frequent)
    post_average(os.path.join(args.output_dir, result_file),
                 os.path.join(args.output_dir, f"LUNAR_{args.data_type}_complex={args.complex}_frequent={args.frequent}_{args.model}.csv"))
    print(f"Finish parsing logs: {args.test_dataset}")
    print(f"Output dir: {args.output_dir}")
