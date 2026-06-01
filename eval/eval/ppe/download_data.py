from huggingface_hub import hf_hub_download
import zipfile

if __name__ == "__main__":

    hf_hub_download(
        "lmarena-ai/ppe-result-data",
        filename="data.zip",
        repo_type="dataset",
        local_dir="./",
    )

    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall("./")
