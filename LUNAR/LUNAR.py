import os
import time
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from functools import partial
from typing import List, Dict, Any

from LUNAR.llm_module.model import InferLLMGrouping
from LUNAR.log_partition.clustering import TopKTokenClustering
from LUNAR.utils import write_json, get_max_retry, verify_template_for_log_regex, validate_template
from LUNAR.utils import preprocess_log_for_query, verify_template_and_update
from LUNAR.template_database import TemplateDatabase


class BaseParser:
    def __init__(self, add_regex, regex, dir_in='./', dir_out='./result/', rex=[],
                 data_type='full', shot=0, cluster_params=None, llm_params=None):
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.df_logs = None
        self.clusters = None
        self.add_regex = add_regex
        self.regex = regex
        self.data_type = data_type
        self.shot = shot
        self.query_count = 0
        self.max_retry_assigned = 2
        self.max_retry = 2
        self.wait_query_time = 0
        self.write_intermediate = True
        self.json_inter_result = []
        self.llm_params = llm_params
        self.cluster_params = cluster_params
        self.flag_update_template = True
        self.template_database = {}

    def parse(self, logName):
        raise NotImplementedError

    def save_results(self, log_name):
        to_path_logs = os.path.join(self.dir_out, f"{log_name}_{self.data_type}.log_structured.csv")
        df_to_save = self.clusters.prepare_save_df()
        df_to_save.to_csv(to_path_logs, index=False)

        to_path_templates = os.path.join(self.dir_out, f"{log_name}_{self.data_type}.log_templates.csv")
        df_templates = df_to_save.loc[:, ['EventId', 'EventTemplate']].drop_duplicates()
        df_templates['EventId_numeric'] = df_templates['EventId'].str.extract('(\d+)').astype(int)
        df_selected_sorted = df_templates.sort_values(by='EventId_numeric')
        df_selected_sorted = df_selected_sorted.drop('EventId_numeric', axis=1)

        df_selected_sorted.to_csv(to_path_templates, index=False)

        # save intermediate results
        if self.write_intermediate:
            to_path_inter = os.path.join(self.dir_out, f"{log_name}_{self.data_type}.log_intermediate.json")
            lookup_table = self.clusters.get_lookup_table()
            _ = [item.update({'template': lookup_table[item["logs_to_query"][0]]}) for item in self.json_inter_result]
            write_json(self.json_inter_result, to_path_inter)

    def get_examplars(self, logs_buckets=None):
        if self.shot == 0:
            examplars = None
        else:
            examplars = [
                {'query': 'try to connected to host: 172.16.254.1, finished.',
                 'answer': 'try to connected to host: {ip_address}, finished.'},
                {'query': 'Search directory: /var/www/html/index.html', 'answer': 'Search directory: {directory}'},
            ]
        return examplars

    def validate_and_update(self, logs_to_query, template):
        time1 = time.time()
        update_success, update_num = False, 0
        if validate_template(template):
            update_success, update_num = self.clusters.update_logs(template)
            print(f"Time for one update logs: {time.time() - time1}, template {template}, ")
        else:
            print(f"Validate template `{template}` failed. Retry query")
        return update_success, update_num

    def validate_and_update_with_cluster_map(self, logs_to_query, template, cluster_id):
        time1 = time.time()
        update_success, update_num = False, 0
        if validate_template(template):
            update_success, update_num, updated_indexes = self.clusters.update_logs_with_map(template, cluster_id)
            print(f"Time for one update logs: {time.time() - time1}, template `{template}`")
        else:
            print(f"Validate template `{template}` failed. Retry query")
        return update_success, update_num

    def validate_and_update_with_cluster_map_template_database(self, logs_to_query, template, cluster_id):
        time1 = time.time()
        update_success, update_num = False, 0
        if validate_template(template):
            update_success, update_num, updated_indexes = self.clusters.update_logs_with_map(template, cluster_id)
            if update_success:
                parent_cluster_id = self.clusters.update_map_child2parent[cluster_id]
                need_update, new_template, insert_indexes = self.template_database[parent_cluster_id].add_template(template, updated_indexes)
                if need_update and validate_template(new_template):
                    update_num = self.clusters.update_logs_by_indexes(new_template, cluster_id, insert_indexes)
                    if new_template != template:
                        _, update_num, updated_indexes = self.clusters.update_logs_with_map(new_template, cluster_id)
                        self.template_database[parent_cluster_id].update_indexes(new_template, updated_indexes)
                        print(f"[TemplateBaseUpdate] Match unparsed logs {update_num} with new template `{new_template}`")
                print(f"Update Success: Time for one update logs: {time.time() - time1}, template `{template}`")
            else:
                print(f"Update failed: Template can not match logs `{template}`. Retry query")
        else:
            print(f"Update failed: Validate template `{template}` failed. Retry query")

        return update_success, update_num


class LUNARParser(BaseParser):
    def __init__(self, add_regex, regex, dir_in='./', dir_out='./result/', rex=[], data_type='full', shot=0,
                 cluster_params=None, llm_params=None):
        super().__init__(add_regex, regex, dir_in, dir_out, rex, data_type, shot, cluster_params, llm_params)
        self.llm = InferLLMGrouping(**self.llm_params)
        if self.cluster_params["cluster_method"] == "TopKToken":
            self.clusters = TopKTokenClustering(sample_method=self.cluster_params["sample_method"],
                                                sample_size=self.cluster_params["sample_size"],
                                                min_cluster_size=self.cluster_params["min_cluster_size"],
                                                cluster_topk=self.cluster_params["cluster_topk"],
                                                sample_min_similarity=self.cluster_params["sample_min_similarity"],
                                                lcu_lamb=self.cluster_params["lcu_lamb"],
                                                lcu_sample_size=self.cluster_params["lcu_sample_size"],
                                                sample_size_auto=self.cluster_params["sample_size_auto"],
                                                add_regex=self.cluster_params["add_regex"],
                                                regex=self.cluster_params["regex"],
                                                pad_query=self.cluster_params["pad_query"])
        else:
            raise NotImplementedError


    def parse(self, logName):
        log_path = os.path.join(self.dir_in, f"{logName}_{self.data_type}.log_structured.csv")
        print('Parsing file: ' + log_path)
        self.clusters.load_data(pd.read_csv(log_path))
        logs_grouped = self.clusters.clustering()
        self.initialize_template_database()

        n_iter = 0
        while self.clusters.num_processed_logs < self.clusters.num_total_logs:
            # Sample logs to query
            print(f"Iteration {n_iter}")
            update_success, logs_to_query, logs_to_query_regex, template, cluster_id = self.parse_one_iter(reparse=False)

            if not update_success:
                retry = 0
                while retry < self.max_retry and not update_success:
                    print(f"Update failed. Retry {retry} times when updating is not successful")
                    update_success, logs_to_query, logs_to_query_regex, template, cluster_id = self.parse_one_iter(reparse=True)
                    retry += 1
                if not update_success:
                    print(f"Update failed. Retry {retry} times failed. Try to get a compromise response")
                    template = self.llm.get_compromise_response(logs_to_query_regex)
                    update_success, update_num = self.validate_and_update_with_cluster_map_template_database(logs_to_query_regex, template, cluster_id)

            if not update_success:
                print(f"Update failed. Retry querying failed. Get a compromise response also failed.")
                update_success, update_num = self.validate_and_update_with_cluster_map_template_database(logs_to_query_regex, logs_to_query_regex[0], cluster_id)
            print("========================================================================================\n\n")
            n_iter += 1
            if len(logs_to_query) > 0:
                save_item = {"iter": n_iter, "logs_to_query": logs_to_query, "logs_to_query_regex": logs_to_query_regex,
                             "llm_template": template, "cluster_id": int(cluster_id),
                             "update_success": update_success,
                             }
                self.json_inter_result.append(save_item)
        self.save_results(logName)

    def parse_parallel(self, logName):
        log_path = os.path.join(self.dir_in, f"{logName}_{self.data_type}.log_structured.csv")
        print('Parsing file: ' + log_path)
        self.clusters.load_data(pd.read_csv(log_path))
        logs_grouped = self.clusters.clustering()

        n_iter = 0
        n_executors = 8

        while self.clusters.num_processed_logs < self.clusters.num_total_logs:
            # Sample logs to query
            print(f"Iteration {n_iter}")
            update_success, logs_to_query, logs_to_query_regex, template, cluster_id = self.parse_one_iter(reparse=False)

            if not update_success:
                retry = 0
                while retry < self.max_retry and not update_success:
                    print(f"Update failed. Retry {retry} times when updating is not successful")
                    update_success, logs_to_query, logs_to_query_regex, template, cluster_id = self.parse_one_iter(reparse=True)
                    retry += 1
                if not update_success:
                    print(f"Update failed. Retry {retry} times failed. Try to get a compromise response")
                    template = self.llm.get_compromise_response(logs_to_query_regex)
                    update_success, update_num = self.validate_and_update_with_cluster_map_template_database(logs_to_query_regex, template, cluster_id)
            if not update_success:
                print(f"Update failed. Retry querying failed. Get a compromise response also failed.")
                update_success, update_num = self.validate_and_update_with_cluster_map_template_database(logs_to_query_regex, logs_to_query[0], cluster_id)
            print("========================================================================================\n\n")
            n_iter += 1
            if len(logs_to_query) > 0:
                save_item = {"iter": n_iter, "logs_to_query": logs_to_query, "logs_to_query_regex": logs_to_query_regex,
                             "llm_template": template, "cluster_id": int(cluster_id),
                             "update_success": update_success,
                             }
                self.json_inter_result.append(save_item)
        self.save_results(logName)

    def parse_one_iter(self, reparse=False):
        cluster_id, logs_to_query = self.clusters.sample_for_llm()
        if self.add_regex == "add":
            logs_to_query_regex = [preprocess_log_for_query(log, self.regex) for log in logs_to_query]
        else:
            logs_to_query_regex = logs_to_query

        # Query LLM
        examplars = self.get_examplars()
        template, query_time = self.llm.parsing_log_templates(logs_to_query_regex, examplars, reparse=reparse)
        self.wait_query_time += query_time
        self.query_count += 1
        print("\t============ Aggregate ====================")
        print("\tAggregated Template: ", template)
        update_success, update_num = self.validate_and_update_with_cluster_map_template_database(logs_to_query_regex, template, cluster_id)

        return update_success, logs_to_query, logs_to_query_regex, template, cluster_id

    def initialize_template_database(self):
        for k, _df in self.clusters.clusters.items():
            cid = _df["cid1"].iloc[0]
            if cid not in self.template_database:
                self.template_database[cid] = TemplateDatabase()


class LUNARParserParallel(BaseParser):
    def __init__(self, add_regex, regex, dir_in='./', dir_out='./result/', rex=[], data_type='full', shot=0,
                 cluster_params=None, llm_params=None):
        super().__init__(add_regex, regex, dir_in, dir_out, rex, data_type, shot, cluster_params, llm_params)
        self.llm = InferLLMGrouping(**self.llm_params)
        if self.cluster_params["cluster_method"] == "TopKToken":
            self.clusters = TopKTokenClustering(sample_method=self.cluster_params["sample_method"],
                                                sample_size=self.cluster_params["sample_size"],
                                                min_cluster_size=self.cluster_params["min_cluster_size"],
                                                cluster_topk=self.cluster_params["cluster_topk"],
                                                sample_min_similarity=self.cluster_params["sample_min_similarity"],
                                                lcu_lamb=self.cluster_params["lcu_lamb"],
                                                lcu_sample_size=self.cluster_params["lcu_sample_size"],
                                                sample_size_auto=self.cluster_params["sample_size_auto"],
                                                add_regex=self.cluster_params["add_regex"],
                                                regex=self.cluster_params["regex"],
                                                pad_query=self.cluster_params["pad_query"])
        else:
            raise NotImplementedError

    def parallel_parse_one_bucket(self, buckets):
        n_iter = 0
        num_processed_logs = 0
        buckets = {k:v for k, v in buckets if len(v) > 0}

        total_bucket_size = sum([len(v) for k, v in buckets.items()])
        while num_processed_logs < total_bucket_size:
            update_success, logs_to_query, logs_to_query_regex, template, cluster_id, updated_buckets = self.parse_one_iter(buckets, reparse=False)
            if not update_success:
                retry = 0
                while retry < self.max_retry and not update_success:
                    print(f"Update failed. Retry {retry} times when updating is not successful")
                    update_success, logs_to_query, logs_to_query_regex, template, cluster_id, updated_buckets = self.parse_one_iter(buckets, reparse=False)
                    retry += 1

                if not update_success:
                    print(f"Update failed. Retry {retry} times failed. Try to get a compromise response")
                    template = self.llm.get_compromise_response(logs_to_query_regex)
                    update_success, update_num, updated_buckets = self.validate_and_update_with_cluster_map_template_database_parallel(logs_to_query_regex, template, cluster_id)
            if not update_success:
                print(f"Update failed. Retry querying failed. Get a compromise response also failed.")
                update_success, update_num, updated_buckets = self.validate_and_update_with_cluster_map_template_database_parallel(logs_to_query_regex, logs_to_query_regex[0], cluster_id)
            print("========================================================================================\n\n")
            n_iter += 1
            if len(logs_to_query) > 0:
                save_item = {"iter": n_iter, "logs_to_query": logs_to_query, "logs_to_query_regex": logs_to_query_regex,
                             "llm_template": template, "cluster_id": int(cluster_id),
                             "update_success": update_success,
                             }
                self.json_inter_result.append(save_item)

    def parse(self, logName):
        pass

    def divide_buckets(self):
        cid_to_bucket = {}
        for k, _df in self.clusters.clusters.items():
            cid = _df["cid1"].iloc[0]
            _df.loc[:, "Matched"] = False
            if cid not in cid_to_bucket:
                cid_to_bucket[cid] = {0: _df}
            else:
                cid_to_bucket[cid][len(cid_to_bucket[cid])] = _df

        buckets = [i[1] for i in sorted(cid_to_bucket.items(), key=lambda x: x[0])]

        return buckets

    def parse_one_iter(self, buckets={}, reparse=False):
        cluster_id, logs_to_query = self.clusters.sample_for_llm(buckets=buckets)
        if self.add_regex == "add":
            logs_to_query_regex = [preprocess_log_for_query(log, self.regex) for log in logs_to_query]
        else:
            logs_to_query_regex = logs_to_query

        # Query LLM
        examplars = self.get_examplars()
        template, query_time = self.llm.parsing_log_templates(logs_to_query_regex, examplars, reparse=reparse)
        self.wait_query_time += query_time
        self.query_count += 1
        print("\t============ Aggregate ====================")
        print("\tAggregated Template: ", template)

        update_success, update_num, updated_buckets = self.validate_and_update_with_cluster_map_template_database_parallel(logs_to_query_regex, template, cluster_id)

        return update_success, logs_to_query, logs_to_query_regex, template, cluster_id, updated_buckets


# =====================================================================================
# ====================== Functions for parallel processing   ==========================
# =====================================================================================
from LUNAR.utils import verify_template_for_log_regex, verify_template_for_log_with_first_token, verify_template_for_log_with_first_token_subset
from LUNAR.log_partition.clustering import remove_duplicates,sampling_from_list, sampling_from_sorted_list, anchor_log_selection

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


def sample_by_lcu_sampling_parallel(input_clusters, sample_size=3, pad_query=False, lcu_lamb=0.6,
                                    lcu_sample_size=3, add_skip_sim=False, sample_min_similarity=100, dedup=True):
    if not input_clusters:
        print("No clusters found")
        return
    # Strategy: always sample the largest cluster
    max_cluster_id = max(input_clusters, key=lambda k: len(input_clusters[k]))
    current_logs_bucket_id = max_cluster_id
    current_logs_bucket = input_clusters[current_logs_bucket_id]
    print(f"Sample {sample_size} from current logs bucket: ID: {current_logs_bucket_id}, Len: {current_logs_bucket['length'].iloc[0]}, Bucket Size: {len(current_logs_bucket)}, Total Buckets: {len(input_clusters)}", )

    if len(current_logs_bucket) == 1:
        logs = current_logs_bucket["Content"].tolist()
        cluster_id = current_logs_bucket["cid2"].iloc[0]
        return cluster_id, sampling_from_list(logs, sample_size, padding=pad_query)
    else:
        anchor_log, candidate_logs = anchor_log_selection(current_logs_bucket["Content"].tolist(),
                                                               method="first")
        if dedup:
            candidate_logs = remove_duplicates(candidate_logs)
        cluster_id = current_logs_bucket["cid2"].iloc[0]
        sampled = sampling_from_sorted_list(anchor_log, candidate_logs, sample_size-1,
                                            method="lcu_sampling", lcu_lamb=lcu_lamb, lcu_sample_size=lcu_sample_size,
                                            add_skip_sim=add_skip_sim,
                                            min_sim_threshold=sample_min_similarity, remove_same=True,
                                            )

        return cluster_id, sampled


def parallel_parse_one_iter(buckets=None, buckets_to_run=None, llm=None, reparse=False, add_regex="no", regex=[]):
    cluster_id, logs_to_query = sample_by_lcu_sampling_parallel(input_clusters=buckets_to_run)
    if add_regex == "add":
        logs_to_query_regex = [preprocess_log_for_query(log, regex) for log in logs_to_query]
    else:
        logs_to_query_regex = logs_to_query

    examplars = None
    template, query_time = llm.parsing_log_templates(logs_to_query_regex, examplars, reparse=reparse)
    # self.wait_query_time += query_time
    # self.query_count += 1
    print("\t============ Aggregate ====================")
    print("\tAggregated Template: ", template)

    # update_success, update_num = self.validate_and_update(logs_to_query, template)
    update_success, update_num, updated_buckets, updated_buckets_to_run = validate_and_update_with_cluster_map_parallel(logs_to_query_regex, template, cluster_id, buckets, buckets_to_run)
    return update_success, logs_to_query, logs_to_query_regex, template, cluster_id, updated_buckets, updated_buckets_to_run


# def parallel_parse_one_bucket(buckets):
def parallel_parse_one_bucket(args):
    llm, buckets = args
    n_iter = 0
    num_processed_logs = 0
    json_inter_result = []
    buckets_to_run = {k: v for k, v in buckets.items() if len(v) > 0}

    total_bucket_size = sum([len(v) for k, v in buckets.items()])
    flag_all_buckets_parsed = all([len(v.loc[v["Matched"] == False]) == 0 for k, v in buckets.items()])
    remaining_logs = sum([len(v.loc[v["Matched"] == False]) for k, v in buckets_to_run.items()])
    while remaining_logs > 0:
        update_success, logs_to_query, logs_to_query_regex, template, cluster_id, buckets, buckets_to_run = parallel_parse_one_iter(buckets, buckets_to_run, llm, reparse=False)
        if not update_success:
            retry = 0
            while retry < args.max_retry and not update_success:
                print(f"Update failed. Retry {retry} times when updating is not successful")
                update_success, logs_to_query, logs_to_query_regex, template, cluster_id, buckets, buckets_to_run = parallel_parse_one_iter(buckets, buckets_to_run, llm, reparse=False)
                retry += 1

            if not update_success:
                print(f"Update failed. Retry {retry} times failed. Try to get a compromise response")
                template = llm.get_compromise_response(logs_to_query_regex)
                update_success, update_num, updated_buckets = validate_and_update_with_cluster_map_parallel(
                    logs_to_query_regex, template, cluster_id, buckets)
        if not update_success:
            print(f"Update failed. Retry querying failed. Get a compromise response also failed.")
            update_success, update_num, updated_buckets = validate_and_update_with_cluster_map_parallel(
                logs_to_query_regex, logs_to_query_regex[0], cluster_id, buckets)
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
# =====================================================================================
# ====================== Functions for parallel processing   ==========================
# =====================================================================================

