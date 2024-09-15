from transformers import BertModel, BertTokenizer

BERT_PATH = './BERT'

bert = BertModel.from_pretrained(BERT_PATH)
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

test_sentence = "She enjoys reading books."
tokens = tokenizer.encode_plus(text=test_sentence, return_tensors='pt')
model_out = bert(**tokens)
print(model_out)

test_sentence_2 = "She loves to read."
tokens_2 = tokenizer.encode_plus(text=test_sentence_2, return_tensors='pt')
model_out_2 = bert(**tokens_2)
print(model_out_2)
