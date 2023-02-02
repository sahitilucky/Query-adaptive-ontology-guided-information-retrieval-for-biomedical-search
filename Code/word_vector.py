import gensim.downloader as api

#info = api.info()  # show info about available models/datasets
model = api.load("glove-wiki-gigaword-300")
model.