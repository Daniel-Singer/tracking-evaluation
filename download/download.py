import sportslabkit as slk 


def download_dataset():
    dl = slk.datasets.KaggleDownloader()
    
    dl.download(force=True)
    