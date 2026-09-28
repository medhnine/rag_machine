from langchain_text_splitters import RecursiveCharacterTextSplitter

doc = f'An intuitive strategy is to split documents based on their length. This simple yet effective approach ensures that each chunk doesn’t exceed a specified size limit. Key benefits of length-based splitting:'
chuncker = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap=20)

print(chuncker.split_text(doc))