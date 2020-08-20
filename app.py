import torch
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

st.title('A.I.Raid')
st.markdown('GPT-2 text generation model trained on Mike Leach quotes')
st.image('https://i.ytimg.com/vi/Wtb1Eqolhzw/maxresdefault.jpg', use_column_width=True)

topic = st.text_input('Suggest a topic', 'pineapple on pizza')

max_length = st.radio(
        "How long do you want Coach Leach's response to be?",
        ('Short','Medium','Long'))
go = st.button('Generate')

@st.cache(allow_output_mutation=True)
def getModel():
        model = GPT2LMHeadModel.from_pretrained('BenDavis71/GPT-2-Finetuning-AIRaid', from_tf=True)
        tokenizer = GPT2Tokenizer.from_pretrained('BenDavis71/GPT-2-Finetuning-AIRaid', from_tf=True)
        return model, tokenizer


def seedSwitch(x, rand):
    switcher = {
        1: f'On the subject of {x},',
        2: f'The problem with America today is that {x}',
        3: f'What do I think about {x}?',
        4: f'Let me tell you about {x}',
        5: f'Everybody keeps saying {x} this, {x} that. Well I think',
        6: f'The thing about {x}:',
        7: f'You know, {x}',
        8: f'It always amazes me how {x}',
        9: f'We used to worry about {x} all the time.'     
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

	rand = np.random.randint(1,9)
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
