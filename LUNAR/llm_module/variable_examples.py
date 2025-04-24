import json

VARIABLE_EXAMPLES_SETTING = {
    "lunar": {
        "regex_from_drain": [],
        "variable_examples": {
            "directory": ["/var/www/html/xxx"],
            "file": ["/var/lib/zookeeper/log.000000001"],
            "blk_id": ["blk_-1234567832142354978"],
            "api": ["com.huawei.health.manager.Service@32a6bf8"],
            "time": ["2017-07-02 15:46:40.536"],
            "ip_or_url": ["192.168.0.1:8008"],
        }
    }
}


def json2prompt(json_response=None):
    """
    Convert JSON response to a string format for variable examples.
    """
    if isinstance(json_response, str):
        json_response = json.loads(json_response)
    if json_response is None:
        var_example = (
            "# Variable Examples: \n"
            "- `/var/www/html/` -> `{directory}`\n"
            "- `worker.jni:onShutdown` -> `{configuration_reference}`\n"
            "- `HTTP/1.1` `HTTP/1.0` -> `{protocol_and_version}`\n"
            "- `blk_1234567832142354978` -> `{blk_id}`\n"
            "- `com.huawei.health.manager.Service@32a6bf8` -> `{service}`\n"
            "- `2017-07-02 15:46:40.536` -> `{time}`\n"
            "- `192.168.0.1:8008` -> `{complex_ip}`\n"
            "- `root` -> `{user}`\n"
        )
    else:
        var_example = "# Variable Examples: \n"
        for k, vars in json_response.items():
            if k == "" or vars == []:
                continue
            k_lower = '_'.join(k.lower().split())
            _vars = [f"`{v}`" for v in vars]
            # line = f"- {', '.join(_vars)} -> `{k_lower}`\n"
            line = f"- {', '.join(_vars)} -> `"+"{"+f"{k_lower}"+"}`\n"
            var_example += line
    return var_example

