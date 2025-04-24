import os
import time
import random
import sys
from multiprocessing import Pool
from functools import partial
import pandas as pd
from LUNAR.log_partition.clustering import compute_adaptive_sample_size
from LUNAR.llm_module.model import InferLLMGrouping
from LUNAR.utils import validate_template, verify_template_for_log_with_first_token, preprocess_log_for_query
from LUNAR.log_partition.clustering import remove_duplicates,sampling_from_list, sampling_from_sorted_list, anchor_log_selection
from LUNAR.LUNAR import LUNARParserParallel
from LUNAR.config import load_args_parallel
from utils.evaluator_main import evaluator, prepare_results, post_average
from LUNAR.template_database import TemplateDatabase

API_KEY = "sk-API_KEY"
BASE_URL = "API_BASE_URL"

llm_params = {
    "model": "gpt-3.5-turbo-0613",
    "base_url": BASE_URL,
    "api_key": API_KEY,
    "dataset": 'all',
    "prompt": "VarExam",
}

llm = InferLLMGrouping(**llm_params)

def validate_and_update_with_cluster_map_parallel(logs_to_query, template, cluster_id, buckets, buckets_to_run):
    time1 = time.time()
    update_success, update_num, updated_buckets, updated_buckets_to_run = False, 0, None, None
    if validate_template(template):
        update_success, update_num, updated_buckets, updated_buckets_to_run = update_logs_with_map_parallel(template, cluster_id, buckets, buckets_to_run)
        print(f"Time for one update logs: {time.time() - time1}, template `{template}`")
        # logs_to_parse = logs_remained
    else:
        print(f"Validate template `{template}` failed. Retry query")
    if not updated_buckets:
        updated_buckets = buckets
        updated_buckets_to_run = buckets_to_run
        print("Tempate not valid, return the original buckets")
    return update_success, update_num, updated_buckets, updated_buckets_to_run


def validate_and_update_with_cluster_map_template_database_parallel(logs_to_query, template, cluster_id, buckets, buckets_to_run, template_database):
    time1 = time.time()
    update_success, update_num, updated_buckets, updated_buckets_to_run = False, 0, None, None
    if validate_template(template):
        update_success, update_num, updated_buckets, updated_buckets_to_run, updated_indexes = update_logs_with_map_template_database_parallel(template, cluster_id, buckets, buckets_to_run)
        if update_success:
            need_update, new_template, insert_indexes = template_database.add_template(template, updated_indexes)
            if need_update and validate_template(new_template):
                total_updated, updated_buckets, updated_buckets_to_run = update_logs_by_indexes(new_template, cluster_id, insert_indexes, updated_buckets, updated_buckets_to_run)
                if new_template != template:
                    _, update_num, updated_buckets, updated_buckets_to_run, updated_indexes = update_logs_with_map_template_database_parallel(new_template, cluster_id, updated_buckets, updated_buckets_to_run)
                    template_database.update_indexes(new_template, updated_indexes)
                    print(f"[TemplateBaseUpdate] Match unparsed logs {update_num} with new template `{new_template}`")

            print(f"Update Success: Time for one update logs: {time.time() - time1}, template `{template}`")
        else:
            print(f"Update failed: Template can not match logs `{template}`. Retry query")
    else:
        print(f"Update failed: Validate template `{template}` failed. Retry query")

    if not updated_buckets:
        updated_buckets = buckets
        updated_buckets_to_run = buckets_to_run
        print("Tempate not valid, return the original buckets")
    return update_success, update_num, updated_buckets, updated_buckets_to_run, template_database


def update_logs_with_map_parallel(template, child_id, all_buckets=None, reduced_buckets=None):
    if not all_buckets:
        return [], 0, all_buckets, reduced_buckets
    if template == "":
        print("Fail to update Template is empty")
        return [], 0, all_buckets, reduced_buckets
    bucket_ids_to_check = list(all_buckets.keys())
    index = []
    total_matched = 0
    total_num_before, total_num_after = 0, 0
    for bucket_id in bucket_ids_to_check:
        current_logs_bucket = reduced_buckets[bucket_id]
        num_berfore = len(current_logs_bucket.loc[current_logs_bucket["Matched"] == False])
        current_logs_bucket.loc[:, "Matched"] = current_logs_bucket.apply(
            lambda row: verify_template_for_log_with_first_token(row["Content"], template), axis=1)
        index = current_logs_bucket[current_logs_bucket["Matched"] == True].index
        # num_processed_logs += len(index)
        current_logs_bucket.loc[index, "Template"] = template

        current_logs_bucket = current_logs_bucket.loc[current_logs_bucket["Matched"] == False]
        reduced_buckets[bucket_id] = current_logs_bucket
        all_buckets[bucket_id].loc[index, "Template"] = template
        num_after = len(current_logs_bucket.loc[current_logs_bucket["Matched"] == False])
        total_matched += num_berfore - num_after
        total_num_before += num_berfore
        total_num_after += num_after
    empty_bucket_num = len([i for i in reduced_buckets.values() if len(i) != 0])
    print(f"[UpdateBucket] Logs: This iter found: {total_matched}")
    print(f"[UpdateBucket] Buckets: Checked {len(bucket_ids_to_check)}, Parent Bucket size: {total_num_before} -> {total_num_after}, remain buckets: {empty_bucket_num}")
    if total_matched == 0:
        return False, 0, all_buckets, reduced_buckets
    return True, len(index), all_buckets, reduced_buckets


def update_logs_with_map_template_database_parallel(template, child_id, all_buckets=None, reduced_buckets=None):
    if not all_buckets:
        return [], 0, all_buckets, reduced_buckets, {}
    if template == "":
        print("Fail to update Template is empty")
        return [], 0, all_buckets, reduced_buckets, {}
    bucket_ids_to_check = list(all_buckets.keys())
    index = []
    all_indexes = {}
    total_matched = 0
    total_num_before, total_num_after = 0, 0
    for bucket_id in bucket_ids_to_check:
        current_logs_bucket = reduced_buckets[bucket_id]
        num_berfore = len(current_logs_bucket.loc[current_logs_bucket["Matched"] == False])
        current_logs_bucket.loc[:, "Matched"] = current_logs_bucket.apply(
            lambda row: verify_template_for_log_with_first_token(row["Content"], template), axis=1)
        index = current_logs_bucket[current_logs_bucket["Matched"] == True].index
        current_logs_bucket.loc[index, "Template"] = template

        current_logs_bucket = current_logs_bucket.loc[current_logs_bucket["Matched"] == False]
        reduced_buckets[bucket_id] = current_logs_bucket
        all_buckets[bucket_id].loc[index, "Template"] = template
        num_after = len(current_logs_bucket.loc[current_logs_bucket["Matched"] == False])
        total_matched += num_berfore - num_after
        total_num_before += num_berfore
        total_num_after += num_after
        all_indexes[bucket_id] = index.tolist()
    empty_bucket_num = len([i for i in reduced_buckets.values() if len(i) != 0])
    print(f"[UpdateBucket] Logs: This iter found: {total_matched}")
    print(f"[UpdateBucket] Buckets: Checked {len(bucket_ids_to_check)}, Parent Bucket size: {total_num_before} -> {total_num_after}, remain buckets: {empty_bucket_num}")
    if total_matched == 0:
        return False, 0, all_buckets, reduced_buckets, {}
    return True, len(index), all_buckets, reduced_buckets, all_indexes


def update_logs_by_indexes(template, child_id, all_indexes, all_buckets=None, reduced_buckets=None):
    if template == "":
        print("[TemplateBaseUpdate] Fail to modify Template from an empty template")
        return 0, {}, {}
    if not all_indexes:
        print("[TemplateBaseUpdate] No existing indexes to check and update")
        return 0, {}, {}
    bucket_ids_to_check = list(all_buckets.keys())
    total, total_updated = 0, 0
    for bucket_id in bucket_ids_to_check:
        # current_logs_bucket = self.clusters[bucket_id]
        current_logs_bucket = all_buckets[bucket_id]
        index = pd.Index(all_indexes[bucket_id])
        try:
            rows_to_process = current_logs_bucket.loc[index]
        except:
            print()
        verify_results = rows_to_process.apply(lambda row: verify_template_for_log_with_first_token(row["Content"], template), axis=1)
        index_to_update = verify_results[verify_results == True].index
        current_logs_bucket.loc[index_to_update, "Template"] = template

        all_buckets[bucket_id] = current_logs_bucket
        # all_buckets[bucket_id].loc[index, "Template"] = template
        total_updated += len(index_to_update)
        total += len(index)
    print(f"[TemplateBaseUpdate] Update previous logs with merged template, succeed/all: {total_updated}/{total}, in child Bucket {bucket_ids_to_check}")
    return total_updated, all_buckets, reduced_buckets


def sample_by_lcu_sampling_parallel(input_clusters, sample_size=3, pad_query=False, lcu_lamb=0.6,
                                    lcu_sample_size=3, add_skip_sim=False, sample_min_similarity=0.5, dedup=True,
                                    sample_size_auto=True, sample_size_assigned=3):
    if not input_clusters:
        print("No clusters found")
        return
    # Strategy: always sample the largest cluster
    max_cluster_id = max(input_clusters, key=lambda k: len(input_clusters[k]))
    current_logs_bucket_id = max_cluster_id
    current_logs_bucket = input_clusters[current_logs_bucket_id]
    print(f"Sample {sample_size} from current logs bucket: ID: {current_logs_bucket_id}, Bucket Size: {len(current_logs_bucket)}, Total Buckets: {len(input_clusters)}", )

    # if len(current_logs_bucket) == 0:
    #     try:
    #         input_clusters.pop(current_logs_bucket_id)
    #         cluster_id, sampled = sample_by_lcu_sampling_parallel(input_clusters, dedup=dedup)
    #     except:
    #         print()
    #     return cluster_id, sampled

    if len(current_logs_bucket) == 1:
        logs = current_logs_bucket["Content"].tolist()
        cluster_id = current_logs_bucket["cid2"].iloc[0]
        return cluster_id, sampling_from_list(logs, 1, padding=pad_query)
    else:
        anchor_log, candidate_logs = anchor_log_selection(current_logs_bucket["Content"].tolist(), method="first")
        if sample_size_auto:
            length_this_bucket = current_logs_bucket["length"].iloc[0]
            sample_size = compute_adaptive_sample_size(length_this_bucket, anchor_log, sample_size_assigned)
        if dedup:
            candidate_logs = remove_duplicates(candidate_logs)
        cluster_id = current_logs_bucket["cid2"].iloc[0]
        sampled = sampling_from_sorted_list(anchor_log, candidate_logs, sample_size-1,
                                            method="lcu_sampling", lcu_lamb=lcu_lamb, lcu_sample_size=lcu_sample_size,
                                            add_skip_sim=add_skip_sim,
                                            min_sim_threshold=sample_min_similarity, remove_same=True,
                                            )

        return cluster_id, sampled


def parallel_parse_one_iter(buckets=None, buckets_to_run=None, template_database=None, reparse=False, add_regex="no", regex=[]):
    cluster_id, logs_to_query = sample_by_lcu_sampling_parallel(input_clusters=buckets_to_run)
    if add_regex == "add":
        logs_to_query_regex = [preprocess_log_for_query(log, regex) for log in logs_to_query]
    else:
        logs_to_query_regex = logs_to_query

    examplars = None
    template, query_time = llm.parsing_log_templates(logs_to_query_regex, examplars, reparse=reparse)
    print("\t============ Aggregate ====================")
    print("\tAggregated Template: ", template)

    update_success, update_num, updated_buckets, updated_buckets_to_run, template_database = validate_and_update_with_cluster_map_template_database_parallel(logs_to_query_regex, template, cluster_id, buckets, buckets_to_run, template_database)
    return update_success, logs_to_query, logs_to_query_regex, template, cluster_id, updated_buckets, updated_buckets_to_run


# def parallel_parse_one_bucket(buckets, args):
def parallel_parse_one_bucket(buckets, add_regex="no", regex=[]):
    template_database = TemplateDatabase()
    n_iter = 0
    num_processed_logs = 0
    json_inter_result = []
    buckets_to_run = {k: v for k, v in buckets.items() if len(v) > 0}

    total_bucket_size = sum([len(v) for k, v in buckets.items()])
    flag_all_buckets_parsed = all([len(v.loc[v["Matched"] == False]) == 0 for k, v in buckets.items()])
    remaining_logs = sum([len(v.loc[v["Matched"] == False]) for k, v in buckets_to_run.items()])
    while remaining_logs > 0:
        update_success, logs_to_query, logs_to_query_regex, template, cluster_id, buckets, buckets_to_run = parallel_parse_one_iter(buckets, buckets_to_run, template_database, reparse=False, add_regex=add_regex, regex=regex)
        if not update_success:
            retry = 0
            while retry < 2 and not update_success:
                print(f"Update failed. Retry {retry} times when updating is not successful")
                update_success, logs_to_query, logs_to_query_regex, template, cluster_id, buckets, buckets_to_run = parallel_parse_one_iter(buckets, buckets_to_run, template_database, reparse=False, add_regex=add_regex, regex=regex)
                retry += 1

            if not update_success:
                print(f"Update failed. Retry {retry} times failed. Try to get a compromise response")
                template = llm.get_compromise_response(logs_to_query_regex)
                update_success, update_num, updated_buckets, updated_buckets_to_run, template_database = validate_and_update_with_cluster_map_template_database_parallel(
                    logs_to_query_regex, template, cluster_id, buckets, buckets_to_run, template_database)

        if not update_success:
            print(f"Update failed. Retry querying failed. Get a compromise response also failed.")
            update_success, update_num, updated_buckets, updated_buckets_to_run, template_database = validate_and_update_with_cluster_map_template_database_parallel(
                logs_to_query_regex, logs_to_query_regex[0], cluster_id, buckets, buckets_to_run, template_database)

        print("========================================================================================\n\n")
        n_iter += 1
        remaining_logs = sum([len(v.loc[v["Matched"] == False]) for k, v in buckets_to_run.items()])
        if len(logs_to_query) > 0:
            save_item = {"iter": n_iter, "logs_to_query": logs_to_query, "logs_to_query_regex": logs_to_query_regex,
                         "llm_template": template, "cluster_id": int(cluster_id),
                         "update_success": update_success,
                         }
            json_inter_result.append(save_item)
    return buckets, json_inter_result


random.seed(22222)
if __name__ == "__main__":
    random.seed(22222)
    print("Start to parse logs")
    print(f"!!!!!!!!!!! llm.dataset: {llm.dataset}")
    args = load_args_parallel()

    input_dataset_dir = os.path.join(args.data_dir, args.test_dataset)
    output_dataset_dir = os.path.join(args.output_dir, args.test_dataset)
    print(f"Input dir: {input_dataset_dir}")
    print(f"Output dir: {output_dataset_dir}")
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)

    time_start = time.time()
    parser = LUNARParserParallel(add_regex=args.add_regex, regex=args.regex, data_type=args.data_type,
                                                    dir_in=input_dataset_dir, dir_out=output_dataset_dir, rex=[],
                                                    llm_params=args.llm_params, shot=args.shot,
                                                    cluster_params=args.cluster_params,
                                                    )

    # Hierarchical Sharding
    log_path = os.path.join(parser.dir_in, f"{args.test_dataset}_{parser.data_type}.log_structured.csv")
    print('Parsing file: ' + log_path)
    parser.clusters.load_data(pd.read_csv(log_path))
    time_start_after_load = time.time()
    _ = parser.clusters.clustering()

    all_buckets = []
    process_num = args.thread
    parallel_buckets = parser.divide_buckets()
    with Pool(processes=process_num) as pool:
        # results = pool.map(parallel_parse_one_bucket, parallel_items)
        annotate_ = partial(
                parallel_parse_one_bucket,
                add_regex=args.add_regex,
                regex=args.regex,
        )
        results = pool.map(annotate_, parallel_buckets)

        buckets_list, json_inter_results_list = zip(*results)
        all_buckets = [v for d in buckets_list for k, v in d.items()]

    final_df = pd.concat(all_buckets)
    final_df = final_df.sort_index()

    parser.write_intermediate = False
    parser.df_logs = final_df
    parser.clusters.df_logs = final_df
    finish_parsing_time = time.time()
    parser.save_results(args.test_dataset)
    print(parser.dir_out)
    output =[]

    print(f"Total parsing time: {finish_parsing_time - time_start_after_load} seconds (no input output)")
    print(f"Total parsing time: {finish_parsing_time - time_start} seconds (no output)")
    print(f"Total parsing time: {time.time() - time_start} seconds (with output)")

    result_file = prepare_results(args.output_dir, otc=args.otc, complex=args.complex, frequent=args.frequent)
    evaluator(args.test_dataset, args.data_type, input_dataset_dir, output_dataset_dir, result_file,
              otc=args.otc, complex=args.complex, frequent=args.frequent)
    post_average(os.path.join(args.output_dir, result_file),
                 os.path.join(args.output_dir, f"LUNAR_{args.data_type}_complex={args.complex}_frequent={args.frequent}_{args.model}.csv"))

