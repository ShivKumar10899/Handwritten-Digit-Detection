import asyncio
import edge_tts
import streamlit as st

async def generate():
    for i in range(10):
        communicate = edge_tts.Communicate(f"Predicted Digit is {str(i)}", "en-US-AriaNeural")
        await communicate.save(f"audio/{i}.mp3")
for i in range(10):
    st.audio(open(f"audio/{i}.mp3","rb").read())
asyncio.run(generate())