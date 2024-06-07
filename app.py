import streamlit as st
import httpx
import asyncio
from httpx import TimeoutException

async def get_streaming_response(query):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream('GET', 'http://localhost:8000/aget_car_information/', params={'query': query}) as response:
            async for chunk in response.aiter_text():
                yield chunk

async def process_query(query):
    # Create a container for the streaming response
    response_container = st.empty()

    # Display a message while waiting for the response
    with st.spinner("Waiting for response..."):
        try:
            # Buffer to store the incoming chunks
            buffer = ""

            # Get the streaming response from FastAPI
            async for chunk in get_streaming_response(query):
                # Append the chunk to the buffer
                buffer += chunk
                response_container.markdown(buffer, unsafe_allow_html=True)

        except TimeoutException:
            pass

def main():
    st.title("Car Manual Generator")

    # Get the query from the user
    query = st.text_input("Enter your question about the car manual:")

    if st.button("Submit"):
        if query:
            # Process the query when the button is clicked
            asyncio.run(process_query(query))
        else:
            st.warning("Please enter a question.")

if __name__ == '__main__':
    main()