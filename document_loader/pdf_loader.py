# need to pip install pypdf  and PyPDFLoader is not good for scanned pdf or complex laylouts there are other 
# pdfloader checout those
# PDFlumberLoader  --> tables/columns
# UnstructuredPDFLoader or AmazonTextratPDFLoader  --> Scanned/image pdfs
# PyMuPDFLoader --> Need layoyt and image data
# relavant Docs : https://docs.langchain.com/oss/python/integrations/document_loaders
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('document_loader/dl-curriculum.pdf')

docs = loader.load()

print(len(docs))



# print(docs[0].page_content)
# print(docs[1].metadata)