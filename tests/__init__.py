from necessary import necessary

with necessary("datasets", soft=True):
    from datasets.utils.logging import disable_progress_bar

    # disable huggingface progress bar when running tests
    disable_progress_bar()
