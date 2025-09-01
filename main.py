import json
from elevenlabs import play
from google import genai
from dotenv import load_dotenv
from google.genai import types
from elevenlabs.client import ElevenLabs

import speech_recognition as sr
import tools
import os
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
conversationHistory = []

# load tools for Jarvis, grabs a json table from tools.json and adds it to funciton declerations
with open('tools.json', 'r') as f:
    tools_dec_data = json.load(f)['functions']

tools_list = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=d['name'],
                description=d['description'],
                parameters_json_schema=d['parameters_json_schema'],
            )
        ]
    ) for d in tools_dec_data
]

functionMap = {
    "get_time": tools.getTime,
    "web_search": tools.web_search,
    "get_weather": tools.get_weather,
    "main_controller_spotify": tools.main_controller_spotify,
    "start_timer": tools.start_timer,
    "search_files": tools.search_files,
    "run_program": tools.run_program,
    "open_tab" : tools.open_tabs,
    "key_control": tools.key_control,
}
#-- end tool initializiation


# Jarvis' brain. the gmemini api logic + garbage because it liked to kill itself. it barley works but who cares
def askgemini(question):
    """
    Sends a question to the Gemini api and stores history
    """
    conversationHistory.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    final_response_text = None
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=conversationHistory,
        config=types.GenerateContentConfig(
            system_instruction="You are an AI assistant named Jarvis like the Jarvis from Iron Man. Address user as 'Sir' when needed. Don't be robotic. Keep your responses short, preferably under 20 words. The date and time after each query is information. Subtly mention it like 'Good evening' or 'You're up late'",
            temperature=0.5,
            tools=tools_list,
        )
    )
    function_handled = False
    function_calls = []
    function_results = []
    for part in response.candidates[0].content.parts:
        if part.function_call:
            function_calls.append(part.function_call)
    if function_calls:
        for function_call in function_calls:
            function_to_call = functionMap.get(function_call.name)
            if function_to_call:
                function_args = getattr(function_call, 'args', {})
                function_result = function_to_call(**function_args)
                function_results.append({
                    "name": function_call.name,
                    "call": function_call,
                    "result": function_result
                })
            else:
                function_results.append({
                    "name": function_call.name,
                    "call": function_call,
                    "result": f"Sir I am sorry but I do not appear to have {function_call.name} available at the moment."
                })
        contents_for_second_call = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=question)]
            ),
            types.Content(
                role="model",
                parts=[{"function_call": fc["call"]} for fc in function_results]
            ),
        ]
        for results in function_results:
            contents_for_second_call.append(
                types.Content(
                    role="function",
                    parts=[types.Part.from_function_response(
                        name=results["name"],
                        response={"result": results["result"]}
                    )]
                )
            )
        final_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents_for_second_call,
            config=types.GenerateContentConfig(
                system_instruction="You are an AI assistant named Jarvis like the Jarvis from Iron Man. Address user as 'Sir' when needed. Don't be robotic. Keep your responses short, preferably under 20 words. The date and time after each query is information. Subtly mention it like 'Good evening' or 'You're up late'",
                temperature=0.5,
                tools=tools_list,
            )
        )

        final_response_text = final_response.text
    else:
        final_response_text = response.text
    conversationHistory.append(types.Content(role="model", parts=[types.Part.from_text(text=final_response_text)]))
    return final_response_text

    # else:
    #     final_response_text=response.text
    #     conversationHistory.append(types.Content(role="model", parts=[types.Part.from_text(text=final_response_text)]))
    #     return final_response_text

#----------TTS SYSTEM---------#
def tts_func(text: str):
    elevenlabs = ElevenLabs(
      api_key=os.getenv("ELEVEN_LABS_KEY_4"),
    )
    audio = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )

    return audio

INITIALIZED = False


#actual process of everything.
def main():
    global INITIALIZED
    while True:
        try:
            if not INITIALIZED:
                print("Jarvis is ready to go!")
                INITIALIZED = True
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio = r.recognize_google(r.listen(source))
            print(f"You: {audio}")
            output = askgemini(audio)
            print(f"Jarvis: {output}")
            play(tts_func(output))
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Earth API; {0}".format(e))
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()