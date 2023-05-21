import os

from necessary import necessary

with necessary("datasets", soft=True):
    from datasets.utils.logging import disable_progress_bar

    # disable huggingface progress bar when running tests
    disable_progress_bar()


os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
