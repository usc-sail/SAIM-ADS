from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

#mbart model and tokenizer declaration 
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load model to device
model.to(device)

#sample text 



