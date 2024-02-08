import torch
import streamlit as st
import transformers
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir='Ingrid0693/mini-bravo'
model = transformers.T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
tokenizer = transformers.T5Tokenizer.from_pretrained(f"{model_dir}")


if __name__=="__main__":   
    def run_model(input_text):
        input_text = str(input_text).replace('\n', '')
        question = "answer: " + ' '.join(input_text.split())
        input_ids = tokenizer(question, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_new_tokens=1024, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, num_return_sequences=5, length_penalty=2, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        for word in response.split():
            yield word + " "
            time.sleep(0.05)

    st.title('Mini-Bravo-01022024')
    st.markdown('Essa é uma versão de testes de conhecimento expandido com 1024 tokens de saída.')
    st.markdown('Idéias de perguntas aceitas:')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('Explain what is meant by economic terms like "market failure".')
    with col2:
        st.markdown('What is Machine Learning?')
    with col3:
        st.markdown('How i can overcome a divorce?')
        

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Write your question here in english..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(run_model(prompt))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        