# read json file
import json

def split_reasoning_traces(json_path: str, reasoning_token: str = "</think>", output_json_file: str | None = None, logger=None):
    with open(json_path, "r") as f:
        data = json.load(f)

        n_without_token = 0

        # for each json in the list split response into reasoning and answer using </think> token
        for item in data["results"]:
            response = item["response"]
            if reasoning_token in response:
                reasoning, answer = response.split(reasoning_token, 1)
                # add reasoning and answer to item
                item["reasoning"] = reasoning.strip()
                item["answer"] = answer.strip()
            else:
                # print(f"No {reasoning_token} token found in response: {response}")
                item["stucked"] = "true"
                n_without_token += 1

        msg = f"Number of responses without {reasoning_token} token: {n_without_token}"
        if logger:
            logger.info(msg)
        else:
            print(msg)

        # save the data to a new json file
        if output_json_file is None:
            output_json_file = json_path.replace(".json", "_splitted.json")
        with open(output_json_file, "w") as f:
            json.dump(data, f, indent=2)

def add_category_to_generation():
    gen_v0_path = "01_prompting/results/svenharms_val_v0_latest_splitted.json"
    gen_v1_path = "01_prompting/results/svenharms_val_v1_latest_splitted.json"
    val_data_path = "00_data/val.json"

    with open(gen_v0_path, "r") as f:
        gen_v0_data = json.load(f)
    with open(gen_v1_path, "r") as f:
        gen_v1_data = json.load(f)
    with open(val_data_path, "r") as f:
        val_data = json.load(f)
    
    for item_v0, item_v1, item_val in zip(gen_v0_data["results"], gen_v1_data["results"], val_data):
        risk_cluster = item_val["risk cluster"]
        item_v0["risk cluster"] = risk_cluster
        item_v1["risk cluster"] = risk_cluster

        risk_category = item_val["risk category"]
        item_v0["risk category"] = risk_category
        item_v1["risk category"] = risk_category
    
    with open(gen_v0_path.replace(".json", "_with_category.json"), "w") as f:
        json.dump(gen_v0_data, f, indent=2)
    with open(gen_v1_path.replace(".json", "_with_category.json"), "w") as f:
        json.dump(gen_v1_data, f, indent=2)

if __name__ == "__main__":
    add_category_to_generation()