# the following line is a magic command
# that will write all the code below it to the python file app.py
# we will then deploy this app.py file on the cloud server where colab is running
# if you have your own server you can just write the code in app.py and deploy it directly


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from operator import itemgetter
import streamlit as st

# Customize initial app landing page
st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("Welcome I am AI Assistant 🤖")

# Manages live updates to a Streamlit app's display by appending new text tokens
# to an existing text stream and rendering the updated text in Markdown
class StreamHandler(BaseCallbackHandler):
  def __init__(self, container, initial_text=""):
    self.container = container
    self.text = initial_text

  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)

# Load a connection to ChatGPT LLM
chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1,
                     streaming=True)

# Add a basic system prompt for LLM behavior
SYS_PROMPT = """
              Act as a helpful assistant and answer questions to the best of your ability.
              Do not make up answers.
              """

# Create a prompt template for langchain to use history to answer user prompts
prompt = ChatPromptTemplate.from_messages(
  [
    ("system", SYS_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
  ]
)

# Create a basic llm chain
llm_chain = (
  prompt
  |
  chatgpt
)

# Store conversation history in Streamlit session state
streamlit_msg_history = StreamlitChatMessageHistory()

# Create a conversation chain
conversation_chain = RunnableWithMessageHistory(
  llm_chain,
  lambda session_id: streamlit_msg_history,  # Accesses memory
  input_messages_key="input",
  history_messages_key="history",
)

# Shows the first message when app starts
if len(streamlit_msg_history.messages) == 0:
  streamlit_msg_history.add_ai_message("How can I help you?")

# Render current messages from StreamlitChatMessageHistory
for msg in streamlit_msg_history.messages:
  st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, display it and show the response
if user_prompt := st.chat_input():
  st.chat_message("human").write(user_prompt)
  # This is where response from the LLM is shown
  with st.chat_message("ai"):
    # Initializing an empty data stream
    stream_handler = StreamHandler(st.empty())
    config = {"configurable": {"session_id": "any"},
              "callbacks": [stream_handler]}
    # Get llm response
    response = conversation_chain.invoke({"input": user_prompt},
                                         config)
