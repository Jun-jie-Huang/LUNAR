OUTPUT_DIR=./saved_results/LUNAR-parallel
if [ "$1" == "all" ]; then
    datasets=("Proxifier" "Apache" "OpenSSH" "HDFS" "OpenStack" "HPC" "Zookeeper" "HealthApp" "Hadoop" "Spark" "BGL" "Linux" "Mac" "Thunderbird")
else
    datasets=("$1")
fi

for dataset in "${datasets[@]}"
do
    echo "Running on all for ${dataset}"
    mkdir -p ${OUTPUT_DIR}/${dataset}
    python main_parallel.py --test_dataset ${dataset} \
                  2>&1 | tee ${OUTPUT_DIR}/${dataset}/log_test.txt
done
