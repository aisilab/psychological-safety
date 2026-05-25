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