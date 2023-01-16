import torch
import streamlit as st
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
import numpy as np

st.title('Bill Walton Chatbot')
st.markdown('GPT-Neo text generation model trained on Bill Walton quotes')
st.image('https://images.squarespace-cdn.com/content/v1/5866ffbad2b857fd0d13208e/1490718360057-NIY9CLEACN6CBGB95XE8/bill-walton.png', use_column_width=True)

topic = st.text_input('Suggest a topic', 'pineapple on pizza')

max_length = st.radio(
        "How long do you want Bill Walton's response to be?",
        ('Short','Medium','Long'))
go = st.button('Generate')

@st.cache(allow_output_mutation=True)
def getModel():
        model = GPTNeoForCausalLM.from_pretrained('BenDavis71/GPT-Neo-Bill-Walton')
        tokenizer = GPT2Tokenizer.from_pretrained('BenDavis71/GPT-Neo-Bill-Walton')
        return model, tokenizer


def seedSwitch(x, rand):
    switcher = {
        1: f'On the subject of {x},',
        2: f'The problem with America today is that {x}',
        3: f'What do I think about {x}?',
        4: f'Let me tell you about {x}',
        5: f'The thing about {x}:',
        6: f'You know, {x}',
        7: f'It always amazes me how {x}',
    }
    
    return switcher.get(rand)

def lengthSwitch(length):
    switcher = {
        'Short': 60,
        'Medium': 150,
        'Long': 270
    }
    
    return switcher.get(length)


if go:
	model, tokenizer = getModel()

	rand = np.random.randint(1,8)
	seed = seedSwitch(topic, rand)
	max_length = lengthSwitch(max_length)
	
	input_ids  = tokenizer.encode(seed, return_tensors = 'pt')
	output = model.generate(
		input_ids, 
		max_length = max_length, 
		temperature=1.0,
		no_repeat_ngram_size=3,
		top_k=40,
		do_sample=True)
	decoded = tokenizer.decode(output[0], skip_special_tokens=True)
	st.write(decoded)
